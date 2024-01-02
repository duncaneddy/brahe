/*!
 * Defines the FileEOPProvider struct and trait for loading
 * and accessing EOP data from files.
 */


use std::fmt;
use std::io;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

use crate::utils::BraheError;

use crate::eop::c04_parser::parse_c04_line;
use crate::eop::eop_provider::EarthOrientationProvider;
use crate::eop::standard_parser::parse_standard_line;
use crate::eop::types::{EOPExtrapolation, EOPType};


// Define a custom key type for the EOP data BTreeMap to enable use
// since f64 does not implement Ord by default
#[derive(Clone, PartialEq, PartialOrd)]
struct EOPKey(f64);

impl Ord for EOPKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

// Technically Eq trait should not be implement for f64, but we are
// certain that the values not be NaN or inf in the input data files
impl Eq for EOPKey {}

/// Stores Earth Orientation Parameters (EOP) data loaded from a file.
///
/// The structure assumes the input data uses the IAU 2010/2000A conventions. That is the
/// precession/nutation parameter values are in terms of `dX` and `dY`, not `dPsi` and `dEps`.
#[derive(Clone)]
pub struct FileEOPProvider {
    /// Internal variable to indicate whether the Earth Orietnation data Object
    /// has been properly initialized
    initialized: bool,
    /// Type of Earth orientation data loaded
    pub eop_type: EOPType,
    /// Primary data structure storing loaded Earth orientation parameter data.
    ///
    /// Key:
    /// - `mjd`: Modified Julian date of the parameter values
    ///
    /// Values:
    /// - `pm_x`: x-component of polar motion correction. Units: (radians)
    /// - `pm_y`: y-component of polar motion correction. Units: (radians)
    /// - `ut1_utc`: Offset of UT1 time scale from UTC time scale. Units: (seconds)
    /// - `dX`: "X" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
    /// - `dY`: "Y" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
    /// - `lod`: Difference between astronomically determined length of day and 86400 second TAI.Units: (seconds)
    ///   day. Units: (seconds)
    data: BTreeMap<EOPKey, (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>)>,
    /// Defines desired behavior for out-of-bounds Earth Orientation data access
    pub extrapolate: EOPExtrapolation,
    /// Defines interpolation behavior of data for requests between data points in table.
    ///
    /// When set to `true` data will be linearly interpolated to the desired time.
    /// When set to `false` data will be given as the value as the closest previous data entry
    /// present.
    pub interpolate: bool,
    /// Minimum date of stored data. This is the value of the smallest key stored in the `data`
    /// HashMap. Value is a modified Julian date.
    pub mjd_min: f64,
    /// Maximum date of stored data. This is the value of the largest key stored in the `data`
    /// HashMap. Behavior
    /// of data retrieval for dates larger than this will be defined by the `extrapolate` value.
    /// Babylon's Fall
    pub mjd_max: f64,
    /// Modified Julian date of last valid Length of Day (LOD) value. Only applicable for
    /// Bulletin A EOP data. Will be 0 for Bulletin B data and the same as `mjd_max` for C04 data.
    pub mjd_last_lod: f64,
    /// Modified Julian date of last valid precession/nutation dX/dY correction values. Only
    /// applicable for Bulletin A. Will always be the sam as `mjd_max` for Bulletin B and C04 data.
    pub mjd_last_dxdy: f64,
}

impl fmt::Display for FileEOPProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FileEOPProvider - type: {}, {} entries, mjd_min: {}, mjd_max: {},  mjd_last_lod: \
        {}, mjd_last_dxdy: {}, extrapolate: {}, \
        interpolate: {}",
            self.eop_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_lod(),
            self.mjd_last_dxdy(),
            self.extrapolate(),
            self.interpolate()
        )
    }
}

impl fmt::Debug for FileEOPProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FileEOPProvider<Type: {}, Length: {}, mjd_min: {}, mjd_max: {},  mjd_last_lod: \
        {}, mjd_last_dxdy: {}, extrapolate: {}, interpolate: {}>",
            self.eop_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_lod(),
            self.mjd_last_dxdy(),
            self.extrapolate(),
            self.interpolate()
        )
    }
}

/// Detects the type of EOP data stored in the given file.
///
/// # Arguments
///
/// * `filepath` - Path to the EOP file
///
/// # Returns
///
/// * `Ok(EOPType)` - Type of EOP data stored in the file
fn detect_eop_file_type(filepath: &Path) -> Result<EOPType, io::Error> {
    // First attempt to open the file and parse as a C04 file
    let file = std::fs::File::open(filepath)?;
    let reader = BufReader::new(file);

    // Read all lines into a vector
    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

    // First check for C04 file header
    if let Some(line) = lines.get(1) {
        if line.contains("C04") {
            // Check if first line of file parses as a C04-type line
            if let Some(line) = lines.get(6) {
                if parse_c04_line(line.to_owned()).is_ok() {
                    return Ok(EOPType::C04);
                }
            }
        }
    }

    // Next test if file parses as a standard-type file
    if let Some(line) = lines.get(0) {
        if parse_standard_line(line.to_owned()).is_ok() {
            return Ok(EOPType::StandardBulletinA);
        }
    }

    Ok(EOPType::Unknown)
}

impl FileEOPProvider {
    pub fn new() -> Self {
        let data: BTreeMap<EOPKey, (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>)> =
            BTreeMap::new();

        Self {
            initialized: false,
            eop_type: EOPType::Unknown,
            data,
            extrapolate: EOPExtrapolation::Zero,
            interpolate: false,
            mjd_min: 0.0,
            mjd_max: 0.0,
            mjd_last_lod: 0.0,
            mjd_last_dxdy: 0.0,
        }
    }

    /// Creates a new FileEOPProvider from a file containing a C04-formatted EOP data file.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the EOP file
    /// * `interpolate` - Whether to interpolate between data points in the EOP file
    /// * `extrapolate` - Defines the behavior for out-of-bounds EOP data access
    ///
    /// # Returns
    ///
    /// * `Ok(FileEOPProvider)` - FileEOPProvider object containing the loaded EOP data
    fn from_c04_file(filepath: &Path, interpolate: bool, extrapolate: EOPExtrapolation) -> Result<Self, BraheError> {
        let file = std::fs::File::open(filepath)?;
        let reader = BufReader::new(file);

        // Creeate main data structures f
        let mut mjd_min = 0.0;
        let mut mjd_max = 0.0;

        let mut data: BTreeMap<EOPKey, (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>)> =
            BTreeMap::new();

        for (line_num, line_str) in reader.lines().enumerate() {
            // There is not header to skip in standard fiesl so we immediately start reading

            let line = match line_str {
                Ok(l) => l,
                Err(e) => {
                    return Err(BraheError::EOPError(format!(
                        "Failed to parse EOP file on line {}: {}",
                        line_num, e
                    )))
                }
            };

            let eop_data = parse_standard_line(line)?;

            let mjd = eop_data.0;

            // Update record or min and max data entry encountered
            // This is kind of hacky since it assumes the EOP data files are sorted,
            // But there are already a number of assumptions on input data formatting.
            if mjd_min == 0.0 {
                mjd_min = mjd;
            }

            if (line_num == 0) || (mjd > mjd_max) {
                mjd_max = mjd;
            }

            data.insert(
                EOPKey(mjd),
                (
                    eop_data.1, eop_data.2, eop_data.3, eop_data.4, eop_data.5, eop_data.6,
                ),
            );
        }

        Ok(Self{
            initialized: true,
            eop_type: EOPType::C04,
            data,
            extrapolate: extrapolate,
            interpolate: interpolate,
            mjd_min: mjd_min,
            mjd_max: mjd_min,
            mjd_last_lod: mjd_max,
            mjd_last_dxdy: mjd_max,
        })
    }

    fn from_standard_file(filepath: &Path, interpolate: bool, extrapolate: EOPExtrapolation) -> Result<Self, BraheError> {
        let file = std::fs::File::open(filepath)?;
        let reader = BufReader::new(file);

        // Creeate main data structures f
        let mut mjd_min = 0.0;
        let mut mjd_max = 0.0;
        let mut mjd_last_lod = 0.0;
        let mut mjd_last_dxdy = 0.0;

        let mut data: BTreeMap<EOPKey, (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>)> =
            BTreeMap::new();

            for (line_num, line_str) in reader.lines().enumerate() {
                // Skip first 14 lines of C04 data file header
                if line_num < 7 {
                    continue;
                }

                let line = match line_str {
                    Ok(l) => l,
                    Err(e) => {
                        return Err(BraheError::EOPError(format!(
                            "Failed to parse EOP file on line {}: {}",
                            line_num, e
                        )))
                    }
                };

                let eop_data = parse_standard_line(line)?;

                let mjd = eop_data.0;

                // Update record or min and max data entry encountered
                // This is kind of hacky since it assumes the EOP data files are sorted,
                // But there are already a number of assumptions on input data formatting.
                if mjd_min == 0.0 {
                    mjd_min = mjd;
                }

                if (line_num == 0) || (mjd > mjd_max) {
                    mjd_max = mjd;
                }

                // Advance last valid MJD of LOD data if Bulletin A and a value was parsed
                if eop_data.6.is_some() {
                    mjd_last_lod = mjd;
                }

                // Advance last valid MJD of dX/dY data if Bulletin A and a value was parsed
                if eop_data.4.is_some() && eop_data.5.is_some() {
                    mjd_last_dxdy = mjd;
                }

                data.insert(
                    EOPKey(mjd),
                    (
                        eop_data.1, eop_data.2, eop_data.3, eop_data.4, eop_data.5, eop_data.6,
                    ),
                );
            }

        Ok(Self{
            initialized: true,
            eop_type: EOPType::C04,
            data,
            extrapolate: extrapolate,
            interpolate: interpolate,
            mjd_min: mjd_min,
            mjd_max: mjd_min,
            mjd_last_lod: mjd_last_lod,
            mjd_last_dxdy: mjd_last_dxdy,
        })
    }

    pub fn from_file(filepath: &Path, interpolate: bool, extrapolate: EOPExtrapolation) -> Result<Self, BraheError> {
        // Detect file type
        match detect_eop_file_type(filepath)? {
            EOPType::C04 => Self::from_c04_file(filepath, interpolate, extrapolate),
            EOPType::StandardBulletinA => Self::from_standard_file(filepath, interpolate, extrapolate),
            _ => Err(BraheError::EOPError(format!(
                "File does not match supported EOP file format: {}",
                filepath.display()
            ))),
        }
    }
}

impl EarthOrientationProvider for FileEOPProvider {
    /// Returns the number of entries in the EOP data structure.
    fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the type of EOP data stored in the data structure.
    fn eop_type(&self) -> EOPType {
        self.eop_type
    }

    /// Returns the extrapolation method used by the EOP data structure.
    fn extrapolate(&self) -> EOPExtrapolation {
        self.extrapolate
    }

    /// Returns whether the EOP data structure supports interpolation.
    fn interpolate(&self) -> bool {
        self.interpolate
    }

    /// Returns the minimum Modified Julian Date (MJD) supported by the EOP data structure.
    fn mjd_min(&self) -> f64 {
        self.mjd_min
    }

    /// Returns the maximum Modified Julian Date (MJD) supported by the EOP data structure.
    fn mjd_max(&self) -> f64 {
        self.mjd_max
    }

    /// Returns the last Modified Julian Date (MJD) supported by the EOP data structure
    /// for which the length of day (LOD) is known.
    fn mjd_last_lod(&self) -> f64 {
        self.mjd_last_lod
    }

    /// Returns the last Modified Julian Date (MJD) supported by the EOP data structure
    /// for which celestial pole offsets (dX, dY) are known.
    fn mjd_last_dxdy(&self) -> f64 {
        self.mjd_last_dxdy
    }

    /// Returns the UT1-UTC offset for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the UT1-UTC offset for.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - UT1-UTC offset in seconds.
    /// * `Err(String)` - Error message if the UT1-UTC offset could not be retrieved.
    fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            let key = EOPKey(mjd);

            if mjd < self.mjd_max {
                if self.interpolate == true {
                    // Get Above and below points
                    let above = self.data.range(key.clone()..).next().unwrap();
                    let below = self.data.range(..key).next_back().unwrap();
    
                    // Time points and values
                    let t1: f64 = below.0.0;
                    let t2: f64 = above.0.0;
                    let y1: f64 = below.1.2;
                    let y2: f64 = above.1.2;
    
                    // Interpolate
                    Ok((y2 - y1) / (t2 - t1) * (mjd - t1) + y1)
                } else {
                    // Get First value below

                    Ok(self.data.range(..key).next_back().unwrap().1.2)
                }
            } else {
                match self.extrapolate {
                    EOPExtrapolation::Zero => Ok(0.0),
                    EOPExtrapolation::Hold => {
                        // UT1-UTC is guaranteed to be present through `mjd_max`
                        Ok(self.data.get(&key).unwrap().2)
                    }
                    EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                        "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                        mjd, self.mjd_max
                    ))),
                }
            }
        } else {
            Err(BraheError::EOPError(String::from("EOP provider not initialized")))
        }
    }
    /// Returns the polar motion (PM) values for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the polar motion (PM) values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64))` - Polar motion (PM) values in radians.
    /// * `Err(String)` - Error message if the polar motion (PM) values could not be retrieved.
    fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        if self.initialized {
            let key = EOPKey(mjd);

            if mjd < self.mjd_max {
                if self.interpolate == true {
                    // Get Above and below points
                    let above = self.data.range(key.clone()..).next().unwrap();
                    let below = self.data.range(..key).next_back().unwrap();
    
                    // Time points and values
                    let t1: f64 = below.0.0;
                    let t2: f64 = above.0.0;
                    let pm_x1: f64 = below.1.0;
                    let pm_x2: f64 = above.1.0;
                    let pm_y1: f64 = below.1.1;
                    let pm_y2: f64 = above.1.1;
    
                    // Interpolate
                    Ok((
                        (pm_x2 - pm_x1) / (t2 - t1) * (mjd - t1) + pm_x1,
                        (pm_y2 - pm_y1) / (t2 - t1) * (mjd - t1) + pm_y1,
                    ))
                } else {
                    // Get First value below
                    let below = self.data.range(..key).next_back().unwrap();
                    Ok((below.1.0, below.1.1))
                }
            } else {
                match self.extrapolate {
                    EOPExtrapolation::Zero => Ok((0.0, 0.0)),
                    EOPExtrapolation::Hold => {
                        // Get Last Value
                        let last = self.data.get(&key);
                        Ok((last.unwrap().0, last.unwrap().1))
                    }
                    EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                        "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                        mjd, self.mjd_max
                    ))),
                }
            }
        } else {
            Err(BraheError::EOPError(String::from("EOP provider not initialized")))
        }
    }

    /// Returns the Celestial Intermediate Pole (CIP) offset values for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the CIP offset values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64))` - CIP offset values in radians.
    /// * `Err(String)` - Error message if the CIP offset values could not be retrieved.
    fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        if self.initialized {
            let key = EOPKey(mjd);

            if mjd < self.mjd_last_dxdy {
                if self.interpolate == true {
                    // Get Above and below points
                    let above = self.data.range(key.clone()..).next().unwrap();
                    let below = self.data.range(..key).next_back().unwrap();
    
                    // Time points and values
                    let t1: f64 = below.0.0;
                    let t2: f64 = above.0.0;
                    let dx1: f64 = below.1.3.unwrap();
                    let dx2: f64 = above.1.3.unwrap();
                    let dy1: f64 = below.1.4.unwrap();
                    let dy2: f64 = above.1.4.unwrap();
    
                    // Interpolate
                    Ok((
                        (dx2 - dx1) / (t2 - t1) * (mjd - t1) + dx1,
                        (dy2 - dy1) / (t2 - t1) * (mjd - t1) + dy1,
                    ))
                } else {
                    // Get First value below
                    let below = self.data.range(..key).next_back().unwrap();
                    Ok((below.1.3.unwrap(), below.1.4.unwrap()))
                }
            } else {
                match self.extrapolate {
                    EOPExtrapolation::Zero => Ok((0.0, 0.0)),
                    EOPExtrapolation::Hold => {
                        // Get last value. This is guaranteed to be present through `mjd_last_dxdy`
                        let last = self.data.get(&key).unwrap();
                        Ok((last.3.unwrap(), last.4.unwrap()))
                    }
                    EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                        "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                        mjd, self.mjd_max
                    ))),
                }
            }
        } else {
            Err(BraheError::EOPError(String::from("EOP provider not initialized")))
        }
    }

    /// Returns the length of day (LOD) value for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the LOD value for.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Length of day (LOD) value in seconds.
    /// * `Err(String)` - Error message if the LOD value could not be retrieved.
    fn get_lod(&self, mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            let key = EOPKey(mjd);

            if mjd < self.mjd_last_lod {
                if self.interpolate == true {
                    // Get Above and below points
                    let above = self.data.range(key.clone()..).next().unwrap();
                    let below = self.data.range(..key).next_back().unwrap();
    
                    // Time points and values
                    let t1: f64 = below.0.0;
                    let t2: f64 = above.0.0;
                    let y1: f64 = below.1.5.unwrap();
                    let y2: f64 = above.1.5.unwrap();
    
                    // Interpolate
                    Ok((y2 - y1) / (t2 - t1) * (mjd - t1) + y1)
                } else {
                    // Get last value.
                    Ok(self.data.range(..key).next_back().unwrap().1.5.unwrap())
                }
            } else {
                match self.extrapolate {
                    EOPExtrapolation::Zero => Ok(0.0),
                    EOPExtrapolation::Hold => {
                        // LOD is guaranteed to be present through `mjd_last_lod`
                        Ok(self.data.get(&key).unwrap().5.unwrap())
                    }
                    EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                        "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                        mjd, self.mjd_max
                    ))),
                }
            }
        } else {
            Err(BraheError::EOPError(String::from("EOP provider not initialized")))
        }
    }

    /// Returns the Earth orientation parameter (EOP) values for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the EOP values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64, f64, f64, f64, f64))` - EOP values.
    /// * `Err(String)` - Error message if the EOP values could not be retrieved.
    #[allow(non_snake_case)]
    fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        let (pm_x, pm_y) = self.get_pm(mjd)?;
        let ut1_utc = self.get_ut1_utc(mjd)?;
        let (dX, dY) = self.get_dxdy(mjd)?;
        let lod = self.get_lod(mjd)?;
        Ok((pm_x, pm_y, ut1_utc, dX, dY, lod))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_detect_eop_file_type() {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir).join("test_assets");

        let c04_file = "EOP_20_C04_one_file_1962-now.txt";
        let standard_file = "finals.all.iau2000.txt";
        let unknown_file = "bad_eop_file.txt";

        assert_eq!(
            detect_eop_file_type(&filepath.clone().join(c04_file)).unwrap(),
            EOPType::C04
        );
        assert_eq!(
            detect_eop_file_type(&filepath.clone().join(standard_file)).unwrap(),
            EOPType::StandardBulletinA
        );
        assert_eq!(
            detect_eop_file_type(&filepath.clone().join(unknown_file)).unwrap(),
            EOPType::Unknown
        );
    }
}
