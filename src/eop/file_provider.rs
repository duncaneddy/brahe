/*!
 * Defines the FileEOPProvider struct and trait for loading
 * and accessing EOP data from files.
 */

// TODO: Add LRU cache to EOP data retrieval to speed up repeated requests

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::io::BufReader;
use std::io::prelude::*;
use std::ops::Bound;
use std::path::Path;

use crate::eop::c04_parser::parse_c04_line;
use crate::eop::eop_provider::EarthOrientationProvider;
use crate::eop::eop_types::{EOPExtrapolation, EOPType};
use crate::eop::standard_parser::parse_standard_line;
use crate::utils::BraheError;

// Package EOP data as part of crate

/// Packaged C04 EOP Data File
static PACKAGED_C04_FILE: &'static [u8] =
    include_bytes!("../../data/eop/EOP_20_C04_one_file_1962-now.txt");

/// Packaged Finals 2000A Data File
static PACKAGED_STANDARD2000_FILE: &'static [u8] =
    include_bytes!("../../data/eop/finals.all.iau2000.txt");

// Define a custom key type for the EOP data BTreeMap to enable use
// since f64 does not implement Ord by default. This is not used
// by and accessors and is only used internally to the FileEOPProvider.
#[derive(Clone, PartialEq, PartialOrd)]
struct EOPKey(f64);

impl Ord for EOPKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

// Technically Eq trait should not be implemented for f64, but we are
// certain that the values not be NaN or inf in the input data files
impl Eq for EOPKey {}

/// Stores Earth Orientation Parameters (EOP) data loaded from a file.
///
/// The structure assumes the input data uses the IAU 2010/2000A conventions. That is the
/// precession/nutation parameter values are in terms of `dX` and `dY`, not `dPsi` and `dEps`.
#[derive(Clone)]
/// Provides Earth Orientation Parameter (EOP) data from a file.
///
/// The `FileEOPProvider` struct represents a provider of Earth Orientation Parameter (EOP) data
/// loaded from a file. It stores the loaded EOP data and provides methods to access and manipulate
/// the data.
///
/// # Fields
///
/// - `initialized`: Internal variable to indicate whether the Earth Orientation data object has been properly initialized.
/// - `eop_type`: Type of Earth orientation data loaded.
/// - `data`: Primary data structure storing loaded Earth orientation parameter data.
/// - `extrapolate`: Defines desired behavior for out-of-bounds Earth Orientation data access.
/// - `interpolate`: Defines interpolation behavior of data for requests between data points in table.
/// - `mjd_min`: Minimum date of stored data.
/// - `mjd_max`: Maximum date of stored data.
/// - `mjd_last_lod`: Modified Julian date of last valid Length of Day (LOD) value.
/// - `mjd_last_dxdy`: Modified Julian date of last valid precession/nutation dX/dY correction values.
///
/// The `data` field is a BTreeMap with the following structure:
///
/// Key:
/// - `mjd`: Modified Julian date of the parameter values
///
/// Values:
/// - `pm_x`: x-component of polar motion correction. Units: (radians)
/// - `pm_y`: y-component of polar motion correction. Units: (radians)
/// - `ut1_utc`: Offset of the UT1 time scale from UTC time scale. Units: (seconds)
/// - `dX`: "X" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
/// - `dY`: "Y" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
/// - `lod`: Difference between astronomically determined length of day and 86400 second TAI.Units: (seconds)
///   day. Units: (seconds)
///
/// The `interpolate` field defines the interpolation behavior of the data for requests between data. If set to `true`
/// the data will be linearly interpolated to the desired time. If set to `false` the data will be given as the value
/// from the closest previous data entry present.
pub struct FileEOPProvider {
    initialized: bool,
    pub eop_type: EOPType,
    data: BTreeMap<EOPKey, (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>)>,
    pub extrapolate: EOPExtrapolation,
    pub interpolate: bool,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub mjd_last_lod: f64,
    pub mjd_last_dxdy: f64,
}

impl fmt::Display for FileEOPProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FileEOPProvider - type: {}, {} entries, mjd_min: {}, mjd_max: {},  mjd_last_lod: \
        {}, mjd_last_dxdy: {}, extrapolation: {}, interpolation: {}",
            self.eop_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_lod(),
            self.mjd_last_dxdy(),
            self.extrapolation(),
            self.interpolation()
        )
    }
}

impl fmt::Debug for FileEOPProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FileEOPProvider<Type: {}, Length: {}, mjd_min: {}, mjd_max: {},  mjd_last_lod: \
        {}, mjd_last_dxdy: {}, extrapolation: {}, interpolation: {}>",
            self.eop_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_lod(),
            self.mjd_last_dxdy(),
            self.extrapolation(),
            self.interpolation()
        )
    }
}

/// Detects the Earth Orientation Parameters (EOP) file type of the given file.
///
/// This function takes a path to a file and returns the detected EOP file type.
/// The file type is determined based on the file's content and the ability for
/// current parsers to read the file.
///
/// # Arguments
///
/// * `filepath` - A `Path` reference to the file whose EOP file type is to be detected.
///
/// # Returns
///
/// * `Ok(EOPType)` - The detected EOP file type if the file type could be successfully determined.
/// * `Err(io::Error)` - An error that occurred while trying to read the file or if the file type could not be determined.
///
/// # Example
///
/// ```ignore
/// use std::env;
/// use std::path::Path;
/// use brahe::eop::file_provider::detect_eop_file_type;
///
/// let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
/// let filepath = Path::new(&manifest_dir)
///                 .join("data")
///                 .join("finals.all.iau2000.txt");
///
/// let filepath = Path::new("path/to/file");
/// let eop_type = detect_eop_file_type(&filepath)?;
/// ```
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
    /// Creates a new FileEOPProvider object. The object is not initialized and cannot be used
    /// until data is loaded into it. This is useful for creating a FileEOPProvider object
    /// that will be loaded with data from a file in the future.
    ///
    /// # Returns
    ///
    /// * `FileEOPProvider` - New FileEOPProvider object.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::FileEOPProvider;
    ///
    /// let eop = FileEOPProvider::new();
    /// ```
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
    /// This should only be used if the file is known to be a C04 file. Otherwise,
    /// the `from_file` function should be used, which can automatically detect the file type.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the EOP file
    /// * `interpolate` - Whether to interpolate between data points in the EOP file
    /// * `extrapolate` - Defines the behavior for out-of-bounds EOP data access
    ///
    /// # Returns
    ///
    /// * `Result<FileEOPProvider, BraheError>` - FileEOPProvider object containing the loaded EOP data, or an Error
    ///
    /// # Example
    ///
    /// ```
    /// use std::env;
    /// use std::path::Path;
    /// use brahe::eop::EOPExtrapolation;
    /// use brahe::eop::FileEOPProvider;
    ///
    /// let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    /// let filepath = Path::new(&manifest_dir)
    ///                 .join("data")
    ///                 .join("eop")
    ///                 .join("EOP_20_C04_one_file_1962-now.txt");
    ///
    /// let eop = FileEOPProvider::from_c04_file(&filepath, true, EOPExtrapolation::Hold).unwrap();
    /// ```
    pub fn from_c04_file(
        filepath: &Path,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        let file = std::fs::File::open(filepath)?;
        let reader = BufReader::new(file);

        Self::from_c04_file_bufreader(reader, interpolate, extrapolate)
    }

    /// Creates a new FileEOPProvider from a BufReader containing a C04-formatted EOP data file.
    /// This is an internal function used to allow for embedding of EOP data files in the package.
    ///
    /// # Arguments
    /// - `reader` - BufReader containing the EOP data file
    ///
    /// # Returns
    /// * `Result<FileEOPProvider, BraheError>` - FileEOPProvider object containing the loaded EOP data, or an Error
    fn from_c04_file_bufreader<T: Read>(
        reader: BufReader<T>,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        // Creeate main data structures f
        let mut mjd_min = 0.0;
        let mut mjd_max = 0.0;

        let mut data: BTreeMap<EOPKey, (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>)> =
            BTreeMap::new();

        for (line_num, line_str) in reader.lines().enumerate() {
            // Skip first 6 lines of C04 data file header
            if line_num < 6 {
                continue;
            }

            let line = match line_str {
                Ok(l) => l,
                Err(e) => {
                    return Err(BraheError::EOPError(format!(
                        "Failed to parse EOP file on line {}: {}",
                        line_num, e
                    )));
                }
            };

            let eop_data = parse_c04_line(line)?;

            let mjd = eop_data.0;

            // Update record or min and max data entry encountered
            // This is kind of hacky since it assumes the EOP data files are sorted,
            // But there are already a number of assumptions on input data formatting.
            if mjd_min == 0.0 {
                mjd_min = mjd;
            }

            if (mjd_max == 0.0) || (mjd > mjd_max) {
                mjd_max = mjd;
            }

            data.insert(
                EOPKey(mjd),
                (
                    eop_data.1, eop_data.2, eop_data.3, eop_data.4, eop_data.5, eop_data.6,
                ),
            );
        }

        Ok(Self {
            initialized: true,
            eop_type: EOPType::C04,
            data,
            extrapolate,
            interpolate,
            mjd_min,
            mjd_max,
            mjd_last_lod: mjd_max,
            mjd_last_dxdy: mjd_max,
        })
    }

    /// Creates a new FileEOPProvider from a file containing a Standard-formatted EOP data file.
    /// This should only be used if the file is known to be a Standard Bulletin A file. Otherwise,
    /// the `from_file` function should be used, which can automatically detect the file type.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the EOP file
    /// * `interpolate` - Whether to interpolate between data points in the EOP file
    /// * `extrapolate` - Defines the behavior for out-of-bounds EOP data access
    ///
    /// # Returns
    ///
    /// * `Result<FileEOPProvider, BraheError>` - FileEOPProvider object containing the loaded EOP data, or an Error
    ///
    /// # Example
    ///
    /// ```
    /// use std::env;
    /// use std::path::Path;
    /// use brahe::eop::EOPExtrapolation;
    /// use brahe::eop::FileEOPProvider;
    ///
    /// let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    /// let filepath = Path::new(&manifest_dir)
    ///                 .join("data")
    ///                 .join("eop")
    ///                 .join("finals.all.iau2000.txt");
    ///
    /// let eop = FileEOPProvider::from_standard_file(&filepath, true, EOPExtrapolation::Hold).unwrap();
    /// ```
    pub fn from_standard_file(
        filepath: &Path,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        let file = std::fs::File::open(filepath)?;
        let reader = BufReader::new(file);

        Self::from_standard_file_bufreader(reader, interpolate, extrapolate)
    }

    /// Creates a new FileEOPProvider from a BufReader containing a Standard-formatted EOP data file.
    /// This is an internal function used to allow for embedding of EOP data files in the package.
    ///
    /// # Arguments
    /// - `reader` - BufReader containing the EOP data file
    /// - `interpolate` - Whether to interpolate between data points in the EOP file
    /// - `extrapolate` - Defines the behavior for out-of-bounds EOP data access
    ///
    /// # Returns
    /// * `Result<FileEOPProvider, BraheError>` - FileEOPProvider object containing the loaded EOP data, or an Error
    fn from_standard_file_bufreader<T: Read>(
        reader: BufReader<T>,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        // Create main data structures f
        let mut mjd_min = 0.0;
        let mut mjd_max = 0.0;
        let mut mjd_last_lod = 0.0;
        let mut mjd_last_dxdy = 0.0;

        let mut data: BTreeMap<EOPKey, (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>)> =
            BTreeMap::new();

        for (line_num, line_str) in reader.lines().enumerate() {
            // There is no header to skip in standard file, so we immediately start reading

            let line = match line_str {
                Ok(l) => l,
                Err(e) => {
                    return Err(BraheError::EOPError(format!(
                        "Failed to parse EOP file on line {}: {}",
                        line_num, e
                    )));
                }
            };

            let eop_data = match parse_standard_line(line) {
                Ok(data) => data,
                Err(e) => {
                    // Skip trying to parse file on first empty pm_x line
                    if e.to_string()
                        .contains("Failed to parse pm_x from '          '")
                    {
                        break;
                    } else {
                        return Err(e);
                    }
                }
            };

            let mjd = eop_data.0;

            // Update record or min and max data entry encountered
            // This is kind of hacky since it assumes the EOP data files are sorted,
            // But there are already a number of assumptions on input data formatting.
            if mjd_min == 0.0 {
                mjd_min = mjd;
            }

            if (mjd_max == 0.0) || (mjd > mjd_max) {
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

        Ok(Self {
            initialized: true,
            eop_type: EOPType::StandardBulletinA,
            data,
            extrapolate,
            interpolate,
            mjd_min,
            mjd_max,
            mjd_last_lod,
            mjd_last_dxdy,
        })
    }

    /// Creates a new FileEOPProvider from a file containing a EOP data file. Function automatically detects
    /// the file type and calls the appropriate function to load the data.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the EOP file
    /// * `interpolate` - Whether to interpolate between data points in the EOP file
    /// * `extrapolate` - Defines the behavior for out-of-bounds EOP data access
    ///
    /// # Returns
    ///
    /// * `Result<FileEOPProvider, BraheError>` - FileEOPProvider object containing the loaded EOP data, or an Error
    ///
    /// # Example
    ///
    /// ```
    /// use std::env;
    /// use std::path::Path;
    /// use brahe::eop::EOPExtrapolation;
    /// use brahe::eop::FileEOPProvider;
    ///
    /// // Load C04 file
    /// let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    /// let filepath = Path::new(&manifest_dir)
    ///                 .join("data")
    ///                 .join("eop")
    ///                 .join("EOP_20_C04_one_file_1962-now.txt");
    ///
    /// let eop = FileEOPProvider::from_file(&filepath, true, EOPExtrapolation::Hold).unwrap();
    ///
    /// // Load Standard Bulletin A file
    /// let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    /// let filepath = Path::new(&manifest_dir)
    ///                 .join("data")
    ///                 .join("eop")
    ///                 .join("finals.all.iau2000.txt");
    ///
    /// let eop = FileEOPProvider::from_file(&filepath, true, EOPExtrapolation::Hold).unwrap();
    /// ```
    pub fn from_file(
        filepath: &Path,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        // Detect file type
        match detect_eop_file_type(filepath)? {
            EOPType::C04 => Self::from_c04_file(filepath, interpolate, extrapolate),
            EOPType::StandardBulletinA => {
                Self::from_standard_file(filepath, interpolate, extrapolate)
            }
            _ => Err(BraheError::EOPError(format!(
                "File does not match supported EOP file format: {}",
                filepath.display()
            ))),
        }
    }

    pub fn from_default_c04(
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        let reader = BufReader::new(PACKAGED_C04_FILE);

        Self::from_c04_file_bufreader(reader, interpolate, extrapolate)
    }

    pub fn from_default_standard(
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        let reader = BufReader::new(PACKAGED_STANDARD2000_FILE);

        Self::from_standard_file_bufreader(reader, interpolate, extrapolate)
    }

    pub fn from_default_file(
        eop_type: EOPType,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        match eop_type {
            EOPType::C04 => Self::from_default_c04(interpolate, extrapolate),
            EOPType::StandardBulletinA => Self::from_default_standard(interpolate, extrapolate),
            _ => Err(BraheError::EOPError(format!(
                "Unsupported EOP file type: {:?}",
                eop_type
            ))),
        }
    }
}

impl EarthOrientationProvider for FileEOPProvider {
    /// Returns the initialization status of the EOP data structure. Value is `true` if the
    /// EOP data structure has been properly initialized and `false` otherwise.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the EOP data structure has been properly initialized, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = FileEOPProvider::from_default_c04(true, EOPExtrapolation::Hold).unwrap();
    /// assert_eq!(eop.is_initialized(), true);
    /// ```
    fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the number of entries stored in the EOP data structure.
    ///
    /// # Returns
    ///
    /// * `usize` - Number of entries stored in the EOP data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = FileEOPProvider::from_default_c04(true, EOPExtrapolation::Hold).unwrap();
    /// assert!(eop.len() >= 18262);
    /// ```
    fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the type of EOP data stored in the data structure. See the `EOPType` enum for
    /// possible values.
    ///
    /// # Returns
    ///
    /// * `EOPType` - Type of EOP data stored in the data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EOPType, EOPExtrapolation, EarthOrientationProvider};
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    /// assert_eq!(eop.eop_type(), EOPType::StandardBulletinA);
    ///
    /// let eop = FileEOPProvider::from_default_c04(true, EOPExtrapolation::Hold).unwrap();
    /// assert_eq!(eop.eop_type(), EOPType::C04);
    /// ```
    fn eop_type(&self) -> EOPType {
        self.eop_type
    }

    /// Returns the extrapolation method used by the EOP data structure. See the `EOPExtrapolation`
    /// enum for possible values.
    ///
    /// # Returns
    ///
    /// * `EOPExtrapolation` - Extrapolation method used by the EOP data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EOPExtrapolation, EarthOrientationProvider};
    ///
    /// // Initialize using Hold extrapolation
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    /// assert_eq!(eop.extrapolation(), EOPExtrapolation::Hold);
    ///
    /// // Initialize using Zero extrapolation
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Zero).unwrap();
    /// assert_eq!(eop.extrapolation(), EOPExtrapolation::Zero);
    ///
    /// // Initialize using Error extrapolation
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Error).unwrap();
    /// assert_eq!(eop.extrapolation(), EOPExtrapolation::Error);
    /// ```
    fn extrapolation(&self) -> EOPExtrapolation {
        self.extrapolate
    }

    /// Returns whether the EOP data structure supports interpolation. If `true`, the data
    /// structure will interpolate between data points when requested. If `false`, the data
    /// structure will return the value from the closest previous data point.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the EOP data structure supports interpolation, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// // Initialize with interpolation enabled
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    /// assert_eq!(eop.interpolation(), true);
    ///
    /// // Calculate midday value intersection manually
    /// let ut1_utc_manual = (eop.get_ut1_utc(58482.0).unwrap() - eop.get_ut1_utc(58481.0).unwrap())/2.0 + eop.get_ut1_utc(58481.0).unwrap();
    ///
    /// // Calculate midday value using provider interpolation
    /// let ut1_utc = eop.get_ut1_utc(58481.5).unwrap();
    ///
    /// // Compare values
    /// assert_eq!(ut1_utc_manual, ut1_utc); // ~ 0.0015388
    ///
    /// // Initialize with interpolation disabled
    /// let eop = FileEOPProvider::from_default_standard(false, EOPExtrapolation::Hold).unwrap();
    /// assert_eq!(eop.interpolation(), false);
    ///
    /// // Confirm hold value for midday is just the previous value in the table
    /// let ut1_utc = eop.get_ut1_utc(eop.mjd_max + 1.0).unwrap();
    /// assert_eq!(ut1_utc, eop.get_ut1_utc(eop.mjd_max).unwrap());
    /// ```
    fn interpolation(&self) -> bool {
        self.interpolate
    }

    /// Returns the minimum Modified Julian Date (MJD) supported by the EOP data structure.
    /// This is the earliest date for which the EOP data structure has data.
    ///
    /// # Returns
    ///
    /// * `f64` - Minimum Modified Julian Date (MJD) supported by the EOP data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    /// assert_eq!(eop.mjd_min(), 41684.0);
    /// ```
    fn mjd_min(&self) -> f64 {
        self.mjd_min
    }

    /// Returns the maximum Modified Julian Date (MJD) supported by the EOP data structure.
    /// This is the latest date for which the EOP data structure has data.
    ///
    /// # Returns
    ///
    /// * `f64` - Maximum Modified Julian Date (MJD) supported by the EOP data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    /// assert!(eop.mjd_max() >= 60679.0);
    /// ```
    fn mjd_max(&self) -> f64 {
        self.mjd_max
    }

    /// Returns the last Modified Julian Date (MJD) supported by the EOP data structure
    /// for which the length of day (LOD) is known.
    ///
    /// # Returns
    ///
    /// * `f64` - Last Modified Julian Date (MJD) for which the length of day (LOD) is known.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    ///
    /// // Confirm last LOD date is after 2022-01-01
    /// assert!(eop.mjd_last_lod() >= 59580.0);
    /// ```
    fn mjd_last_lod(&self) -> f64 {
        self.mjd_last_lod
    }

    /// Returns the last Modified Julian Date (MJD) supported by the EOP data structure
    /// for which celestial pole offsets (dX, dY) are known.
    ///
    /// # Returns
    ///
    /// * `f64` - Last Modified Julian Date (MJD) for which celestial pole offsets (dX, dY) are known.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    ///
    /// // Confirm last dX/dY date is after 2022-01-01
    /// assert!(eop.mjd_last_dxdy() >= 59580.0);
    /// ```
    fn mjd_last_dxdy(&self) -> f64 {
        self.mjd_last_dxdy
    }

    /// Returns the UT1-UTC offset for the given Modified Julian Date (MJD). Return value depends on
    /// the `interpolate` and `extrapolate` settings of the EOP data structure.
    ///
    /// Setting `interpolate` to `true` will cause the data structure to linearly interpolate the returned
    /// value between data points. Setting `interpolate` to `false` will cause the data structure to return
    /// the value from the closest previous data point.
    ///
    /// Setting `extrapolate` to `EOPExtrapolation::Zero` will cause the data structure to return a value of
    /// zero for any request beyond the end of the loaded data. Setting `extrapolate` to `EOPExtrapolation::Hold`
    /// will cause the data structure to return the last valid value for any request beyond the end of the loaded data.
    /// Setting `extrapolate` to `EOPExtrapolation::Error` will cause the data structure to return an error for any
    /// request beyond the end of the loaded data.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the UT1-UTC offset for.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - UT1-UTC offset in seconds.
    /// * `Err(String)` - Error message if the UT1-UTC offset could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    ///
    /// // Confirm UT1-UTC value for 2022-01-01
    /// assert!((eop.get_ut1_utc(59580.0).unwrap() - -0.1104988).abs() < 1e-6);
    /// ```
    fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            if mjd < self.mjd_max {
                if self.interpolate == true {
                    // Get cursor pointing at the gap after the data for the previous data point
                    let cursor = self.data.lower_bound(Bound::Included(&EOPKey(mjd)));


                    // Time points and values
                    let (t1, data1) = cursor.peek_prev().unwrap();
                    let (t2, data2) = cursor.peek_next().unwrap();

                    let t1 = t1.0;
                    let t2 = t2.0;
                    let y1 = data1.2;
                    let y2 = data2.2;

                    // Interpolate, checking if we are exactly at a data point
                    if t1 == t2 {
                        Ok(y1)
                    } else {
                        Ok((y2 - y1) / (t2 - t1) * (mjd - t1) + y1)
                    }
                } else {
                    // Get First value below - Note the "upper" bound is actually the lower time value
                    Ok(self
                        .data
                        .lower_bound(Bound::Included(&EOPKey(mjd)))
                        .peek_prev()
                        .unwrap().1.2)
                }
            } else {
                match self.extrapolate {
                    EOPExtrapolation::Zero => Ok(0.0),
                    EOPExtrapolation::Hold => {
                        // UT1-UTC is guaranteed to be present through `mjd_max`
                        Ok(self.data.get(&EOPKey(self.mjd_max)).unwrap().2)
                    }
                    EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                        "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                        mjd, self.mjd_max
                    ))),
                }
            }
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }

    /// Returns the polar motion (PM) values for the given Modified Julian Date (MJD). Return value depends on
    /// the `interpolate` and `extrapolate` settings of the EOP data structure.
    ///
    /// Setting `interpolate` to `true` will cause the data structure to linearly interpolate the returned
    /// value between data points. Setting `interpolate` to `false` will cause the data structure to return
    /// the value from the closest previous data point.
    ///
    /// Setting `extrapolate` to `EOPExtrapolation::Zero` will cause the data structure to return a value of
    /// zero for any request beyond the end of the loaded data. Setting `extrapolate` to `EOPExtrapolation::Hold`
    /// will cause the data structure to return the last valid value for any request beyond the end of the loaded data.
    /// Setting `extrapolate` to `EOPExtrapolation::Error` will cause the data structure to return an error for any
    /// request beyond the end of the loaded data.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the polar motion (PM) values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64))` - Polar motion (PM) values in radians.
    /// * `Err(String)` - Error message if the polar motion (PM) values could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    /// use brahe::constants::AS2RAD;
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    ///
    /// // Confirm polar motion values for 2022-01-01
    /// assert!((eop.get_pm(59580.0).unwrap().0 - 0.054644 * AS2RAD).abs() < 1e-6);
    /// assert!((eop.get_pm(59580.0).unwrap().1 - 0.276986 * AS2RAD).abs() < 1e-6);
    /// ```
    fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        if self.initialized {
            if mjd < self.mjd_max {
                if self.interpolate == true {
                    // Get cursor pointing at the gap after the data for the previous data point
                    let cursor = self.data.lower_bound(Bound::Included(&EOPKey(mjd)));


                    // Time points and values
                    let (t1, data1) = cursor.peek_prev().unwrap();
                    let (t2, data2) = cursor.peek_next().unwrap();

                    let t1 = t1.0;
                    let t2 = t2.0;
                    let pm_x1 = data1.0;
                    let pm_x2 = data2.0;
                    let pm_y1 = data1.1;
                    let pm_y2 = data2.1;

                    // Interpolate
                    if t1 == t2 {
                        Ok((pm_x1, pm_y1))
                    } else {
                        Ok((
                            (pm_x2 - pm_x1) / (t2 - t1) * (mjd - t1) + pm_x1,
                            (pm_y2 - pm_y1) / (t2 - t1) * (mjd - t1) + pm_y1,
                        ))
                    }
                } else {
                    // Get First value below - Note the "upper" bound is actually the lower time value
                    let below = self.data.lower_bound(Bound::Included(&EOPKey(mjd))).peek_prev().unwrap().1;
                    Ok((below.0, below.1))
                }
            } else {
                match self.extrapolate {
                    EOPExtrapolation::Zero => Ok((0.0, 0.0)),
                    EOPExtrapolation::Hold => {
                        // Get Last Value
                        let last = self.data.get(&EOPKey(self.mjd_max));
                        Ok((last.unwrap().0, last.unwrap().1))
                    }
                    EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                        "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                        mjd, self.mjd_max
                    ))),
                }
            }
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }

    /// Returns the Celestial Intermediate Pole (CIP) offset values for the given Modified Julian Date (MJD). Return value depends on
    /// the `interpolate` and `extrapolate` settings of the EOP data structure.
    ///
    /// Setting `interpolate` to `true` will cause the data structure to linearly interpolate the returned
    /// value between data points. Setting `interpolate` to `false` will cause the data structure to return
    /// the value from the closest previous data point.
    ///
    /// Setting `extrapolate` to `EOPExtrapolation::Zero` will cause the data structure to return a value of
    /// zero for any request beyond the end of the loaded data. Setting `extrapolate` to `EOPExtrapolation::Hold`
    /// will cause the data structure to return the last valid value for any request beyond the end of the loaded data.
    /// Setting `extrapolate` to `EOPExtrapolation::Error` will cause the data structure to return an error for any
    /// request beyond the end of the loaded data.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the Celestial Intermediate Pole (CIP) offset values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64))` - CIP offset values in radians.
    /// * `Err(String)` - Error message if the CIP offset values could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    /// use brahe::constants::AS2RAD;
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    ///
    /// // Confirm CIP offset values for 2022-01-01
    /// assert!((eop.get_dxdy(59580.0).unwrap().0 - 0.095 * 1.0e-3 * AS2RAD).abs() < 1e-6);
    /// assert!((eop.get_dxdy(59580.0).unwrap().1 - -0.250 * 1.0e-3 * AS2RAD).abs() < 1e-6);
    /// ```
    fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        if self.initialized {
            if mjd < self.mjd_last_dxdy {
                if self.interpolate == true {
                    // Get cursor pointing at the gap after the data for the previous data point
                    let cursor = self.data.lower_bound(Bound::Included(&EOPKey(mjd)));


                    // Time points and values
                    let (t1, data1) = cursor.peek_prev().unwrap();
                    let (t2, data2) = cursor.peek_next().unwrap();

                    let t1 = t1.0;
                    let t2 = t2.0;
                    let dx1 = data1.3.unwrap();
                    let dx2 = data2.3.unwrap();
                    let dy1 = data1.4.unwrap();
                    let dy2 = data2.4.unwrap();

                    // Interpolate
                    if t1 == t2 {
                        Ok((dx1, dy1))
                    } else {
                        Ok((
                            (dx2 - dx1) / (t2 - t1) * (mjd - t1) + dx1,
                            (dy2 - dy1) / (t2 - t1) * (mjd - t1) + dy1,
                        ))
                    }
                } else {
                    // Get First value below - Note the "upper" bound is actually the lower time value
                    let below = self.data.lower_bound(Bound::Included(&EOPKey(mjd))).peek_prev().unwrap().1;
                    Ok((
                        below.3.unwrap(),
                        below.4.unwrap(),
                    ))
                }
            } else {
                match self.extrapolate {
                    EOPExtrapolation::Zero => Ok((0.0, 0.0)),
                    EOPExtrapolation::Hold => {
                        // Get last value. This is guaranteed to be present through `mjd_last_dxdy`
                        let last = self.data.get(&EOPKey(self.mjd_last_dxdy)).unwrap();
                        Ok((last.3.unwrap(), last.4.unwrap()))
                    }
                    EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                        "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                        mjd, self.mjd_max
                    ))),
                }
            }
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }

    /// Returns the length of day (LOD) value for the given Modified Julian Date (MJD). Return value depends on
    /// the `interpolate` and `extrapolate` settings of the EOP data structure.
    ///
    /// Setting `interpolate` to `true` will cause the data structure to linearly interpolate the returned
    /// value between data points. Setting `interpolate` to `false` will cause the data structure to return
    /// the value from the closest previous data point.
    ///
    /// Setting `extrapolate` to `EOPExtrapolation::Zero` will cause the data structure to return a value of
    /// zero for any request beyond the end of the loaded data. Setting `extrapolate` to `EOPExtrapolation::Hold`
    /// will cause the data structure to return the last valid value for any request beyond the end of the loaded data.
    /// Setting `extrapolate` to `EOPExtrapolation::Error` will cause the data structure to return an error for any
    /// request beyond the end of the loaded data.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the length of day (LOD) value for.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Length of day (LOD) value in seconds.
    /// * `Err(String)` - Error message if the LOD value could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    ///
    /// // Confirm LOD value for 2022-01-01
    /// assert!((eop.get_lod(59580.0).unwrap() - -0.0267 * 1.0e-3).abs() < 1e-6);
    /// ```
    fn get_lod(&self, mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            if mjd < self.mjd_last_lod {
                if self.interpolate == true {
                    // Get cursor pointing at the gap after the data for the previous data point
                    let cursor = self.data.lower_bound(Bound::Included(&EOPKey(mjd)));


                    // Time points and values
                    let (t1, data1) = cursor.peek_prev().unwrap();
                    let (t2, data2) = cursor.peek_next().unwrap();

                    let t1 = t1.0;
                    let t2 = t2.0;
                    let y1 = data1.5.unwrap();
                    let y2 = data2.5.unwrap();

                    // Interpolate
                    if t1 == t2 {
                        Ok(y1)
                    } else {
                        Ok((y2 - y1) / (t2 - t1) * (mjd - t1) + y1)
                    }
                } else {
                    // Get last value below - Note the "upper" bound is actually the lower time value
                    Ok(self
                        .data
                        .lower_bound(Bound::Included(&EOPKey(mjd)))
                        .peek_prev()
                        .unwrap()
                        .1.5
                        .unwrap())
                }
            } else {
                match self.extrapolate {
                    EOPExtrapolation::Zero => Ok(0.0),
                    EOPExtrapolation::Hold => {
                        // LOD is guaranteed to be present through `mjd_last_lod`
                        Ok(self.data.get(&EOPKey(self.mjd_last_lod)).unwrap().5.unwrap())
                    }
                    EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                        "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                        mjd, self.mjd_max
                    ))),
                }
            }
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }

    /// Returns the full set of Earth orientation parameter (EOP) values for the given Modified Julian Date (MJD).
    /// Return value depends on the `interpolate` and `extrapolate` settings of the EOP data structure.
    ///
    /// Setting `interpolate` to `true` will cause the data structure to linearly interpolate the returned
    /// value between data points. Setting `interpolate` to `false` will cause the data structure to return
    /// the value from the closest previous data point.
    ///
    /// Setting `extrapolate` to `EOPExtrapolation::Zero` will cause the data structure to return a value of
    /// zero for any request beyond the end of the loaded data. Setting `extrapolate` to `EOPExtrapolation::Hold`
    /// will cause the data structure to return the last valid value for any request beyond the end of the loaded data.
    /// Setting `extrapolate` to `EOPExtrapolation::Error` will cause the data structure to return an error for any
    /// request beyond the end of the loaded data.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the EOP values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64, f64, f64, f64, f64))` - EOP values.
    /// * `Err(String)` - Error message if the EOP values could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{FileEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    /// use brahe::constants::AS2RAD;
    ///
    /// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    /// let (pm_x, pm_y, ut1_utc, dX, dY, lod) = eop.get_eop(59580.0).unwrap();
    ///
    /// // Confirm EOP values for 2022-01-01
    /// assert!((pm_x - 0.054644 * AS2RAD).abs() < 1e-6);
    /// assert!((pm_y - 0.276986 * AS2RAD).abs() < 1e-6);
    /// assert!((ut1_utc - -0.1104988).abs() < 1e-6);
    /// assert!((dX - 0.095 * 1.0e-3 * AS2RAD).abs() < 1e-6);
    /// assert!((dY - -0.250 * 1.0e-3 * AS2RAD).abs() < 1e-6);
    /// assert!((lod - -0.0267 * 1.0e-3).abs() < 1e-6);
    /// ```
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
    use std::env;

    use approx::assert_abs_diff_eq;

    use crate::constants::AS2RAD;

    use super::*;

    fn setup_test_eop(
        eop_interpolation: bool,
        eop_extrapolation: EOPExtrapolation,
    ) -> FileEOPProvider {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("finals.all.iau2000.txt");

        let eop_result =
            FileEOPProvider::from_file(&filepath, eop_interpolation, eop_extrapolation);
        assert_eq!(eop_result.is_err(), false);
        let eop = eop_result.unwrap();
        assert_eq!(eop.initialized, true);

        eop
    }

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

    #[test]
    fn test_from_c04_file() {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("EOP_20_C04_one_file_1962-now.txt");

        let eop = FileEOPProvider::from_file(&filepath, true, EOPExtrapolation::Hold).unwrap();

        assert!(eop.is_initialized());
        assert_eq!(eop.len(), 22605);
        assert_eq!(eop.mjd_min(), 37665.0);
        assert_eq!(eop.mjd_max(), 60269.0);
        assert_eq!(eop.eop_type(), EOPType::C04);
        assert_eq!(eop.extrapolation(), EOPExtrapolation::Hold);
        assert_eq!(eop.interpolation(), true);
    }

    #[test]
    fn test_from_default_c04() {
        let eop =
            FileEOPProvider::from_default_file(EOPType::C04, true, EOPExtrapolation::Hold).unwrap();

        // These need to be structured slightly differently since the
        // default package data is regularly updated.
        assert!(eop.is_initialized());
        assert_ne!(eop.len(), 0);
        assert_eq!(eop.mjd_min(), 37665.0);
        assert!(eop.mjd_max() >= 60269.0);
        assert_eq!(eop.eop_type(), EOPType::C04);
        assert_eq!(eop.extrapolation(), EOPExtrapolation::Hold);
        assert_eq!(eop.interpolation(), true);
    }

    #[test]
    fn test_from_standard_file() {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("finals.all.iau2000.txt");

        let eop = FileEOPProvider::from_file(&filepath, true, EOPExtrapolation::Hold).unwrap();

        assert!(eop.is_initialized());
        assert_eq!(eop.len(), 18989);
        assert_eq!(eop.mjd_min(), 41684.0);
        assert_eq!(eop.mjd_max(), 60672.0);
        assert_eq!(eop.eop_type(), EOPType::StandardBulletinA);
        assert_eq!(eop.extrapolation(), EOPExtrapolation::Hold);
        assert_eq!(eop.interpolation(), true);
    }

    #[test]
    fn test_from_default_standard() {
        let eop = FileEOPProvider::from_default_file(
            EOPType::StandardBulletinA,
            true,
            EOPExtrapolation::Hold,
        )
            .unwrap();

        // These need to be structured slightly differently since the
        // default package data is regularly updated.
        assert!(eop.is_initialized());
        assert_ne!(eop.len(), 0);
        assert_eq!(eop.mjd_min(), 41684.0);
        assert!(eop.mjd_max() >= 60672.0);
        assert_eq!(eop.eop_type(), EOPType::StandardBulletinA);
        assert_eq!(eop.extrapolation(), EOPExtrapolation::Hold);
        assert_eq!(eop.interpolation(), true);
    }

    #[test]
    fn test_get_ut1_utc() {
        let eop = setup_test_eop(true, EOPExtrapolation::Hold);

        // Test getting exact point in table
        let ut1_utc = eop.get_ut1_utc(59569.0).unwrap();
        assert_eq!(ut1_utc, -0.1079939);

        // Test interpolating within table
        let ut1_utc = eop.get_ut1_utc(59569.5).unwrap();
        assert_eq!(ut1_utc, (-0.1079939 + -0.1075984) / 2.0);

        // Test extrapolation hold
        let ut1_utc = eop.get_ut1_utc(99999.0).unwrap();
        assert_eq!(ut1_utc, 0.0420038);

        // Test extrapolation zero
        let eop = setup_test_eop(true, EOPExtrapolation::Zero);

        let ut1_utc = eop.get_ut1_utc(99999.0).unwrap();
        assert_eq!(ut1_utc, 0.0);

        // Test return without interpolation
        let eop = setup_test_eop(false, EOPExtrapolation::Hold);

        let ut1_utc = eop.get_ut1_utc(59569.5).unwrap();
        assert_eq!(ut1_utc, -0.1079939);
    }

    #[test]
    fn test_get_pm_xy() {
        let eop = setup_test_eop(true, EOPExtrapolation::Hold);

        // Test getting exact point in table
        let (pm_x, pm_y) = eop.get_pm(59569.0).unwrap();
        assert_eq!(pm_x, 0.075382 * AS2RAD);
        assert_eq!(pm_y, 0.263451 * AS2RAD);

        // Test interpolating within table
        let (pm_x, pm_y) = eop.get_pm(59569.5).unwrap();
        assert_eq!(pm_x, (0.075382 * AS2RAD + 0.073157 * AS2RAD) / 2.0);
        assert_eq!(pm_y, (0.263451 * AS2RAD + 0.264273 * AS2RAD) / 2.0);

        // Test extrapolation hold
        let (pm_x, pm_y) = eop.get_pm(99999.0).unwrap();
        assert_eq!(pm_x, 0.173369 * AS2RAD);
        assert_eq!(pm_y, 0.266914 * AS2RAD);

        // Test extrapolation zero
        let eop = setup_test_eop(true, EOPExtrapolation::Zero);

        let (pm_x, pm_y) = eop.get_pm(99999.0).unwrap();
        assert_eq!(pm_x, 0.0);
        assert_eq!(pm_y, 0.0);

        // Test return without interpolation
        let eop = setup_test_eop(false, EOPExtrapolation::Hold);

        let (pm_x, pm_y) = eop.get_pm(59569.5).unwrap();
        assert_eq!(pm_x, 0.075382 * AS2RAD);
        assert_eq!(pm_y, 0.263451 * AS2RAD);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_get_dxdy() {
        let eop = setup_test_eop(true, EOPExtrapolation::Hold);

        // Test getting exact point in table
        let (dX, dY) = eop.get_dxdy(59569.0).unwrap();
        assert_eq!(dX, 0.265 * 1.0e-3 * AS2RAD);
        assert_eq!(dY, -0.067 * 1.0e-3 * AS2RAD);

        // Test interpolating within table
        let (dX, dY) = eop.get_dxdy(59569.5).unwrap();
        assert_eq!(dX, (0.265 * AS2RAD + 0.268 * AS2RAD) * 1.0e-3 / 2.0);
        assert_abs_diff_eq!(
            dY,
            (-0.067 * AS2RAD + -0.067 * AS2RAD) * 1.0e-3 / 2.0,
            epsilon = f64::EPSILON
        );

        // Test extrapolation hold
        let (dX, dY) = eop.get_dxdy(99999.0).unwrap();
        assert_eq!(dX, 0.006 * 1.0e-3 * AS2RAD);
        assert_eq!(dY, -0.118 * 1.0e-3 * AS2RAD);

        // Test extrapolation zero
        let eop = setup_test_eop(true, EOPExtrapolation::Zero);

        let (dX, dY) = eop.get_dxdy(99999.0).unwrap();
        assert_eq!(dX, 0.0);
        assert_eq!(dY, 0.0);

        // Test return without interpolation
        let eop = setup_test_eop(false, EOPExtrapolation::Hold);

        let (dX, dY) = eop.get_dxdy(59569.5).unwrap();
        assert_eq!(dX, 0.265 * 1.0e-3 * AS2RAD);
        assert_eq!(dY, -0.067 * 1.0e-3 * AS2RAD);
    }

    #[test]
    fn test_get_lod() {
        let eop = setup_test_eop(true, EOPExtrapolation::Hold);

        // Test getting exact point in table
        let lod = eop.get_lod(59569.0).unwrap();
        assert_eq!(lod, -0.3999 * 1.0e-3);

        // Test interpolating within table
        let lod = eop.get_lod(59569.5).unwrap();
        assert_eq!(lod, (-0.3999 + -0.3604) * 1.0e-3 / 2.0);

        // Test extrapolation hold
        let lod = eop.get_lod(99999.0).unwrap();
        assert_eq!(lod, 0.7706 * 1.0e-3);

        // Test extrapolation zero
        let eop = setup_test_eop(true, EOPExtrapolation::Zero);

        let lod = eop.get_lod(99999.0).unwrap();
        assert_eq!(lod, 0.0);

        // Test return without interpolation
        let eop = setup_test_eop(false, EOPExtrapolation::Hold);

        let lod = eop.get_lod(59569.5).unwrap();
        assert_eq!(lod, -0.3999 * 1.0e-3);
    }

    #[test]
    fn test_eop_extrapolation_error() {
        let eop = setup_test_eop(true, EOPExtrapolation::Error);

        // UT1-UTC
        assert!(eop.get_ut1_utc(99999.0).is_err());

        // Polar Motion
        assert!(eop.get_pm(99999.0).is_err());

        // dX, dY
        assert!(eop.get_dxdy(99999.0).is_err());

        // LOD
        assert!(eop.get_lod(99999.0).is_err());
    }

    // TODO: Fix this test
    // #[test]
    // #[allow(non_snake_case)]
    // fn test_cip_format_consistency() {
    //     // Check that the units of C04 and Standard format CIP corrections are
    //     // approximately the same

    //     // Load Standard file
    //     let eop_standard = EarthOrientationProvider::new();

    //     let _eop_standard_result = eop_standard
    //         .from_standard_file(
    //             filepath.to_str().unwrap(),
    //             EOPExtrapolation::Hold,
    //             true,
    //             EOPType::StandardBulletinA,
    //         )
    //         .unwrap();
    //     assert!(eop_standard.is_initialized());

    //     // Load C04 file
    //     let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    //     let filepath = Path::new(&manifest_dir)
    //         .join("test_assets")
    //         .join("iau2000A_c04_14.txt");

    //     let eop_c04 = EarthOrientationProvider::new();

    //     let _eop_c04_result = eop_c04
    //         .from_c04_file(filepath.to_str().unwrap(), EOPExtrapolation::Hold, true)
    //         .unwrap();
    //     assert!(eop_c04.is_initialized());

    //     // Confirm xp and yp are approximately equal
    //     let (pm_x_s, pm_y_s) = eop_standard.get_pm(54195.0).unwrap();
    //     let (pm_x_c04, pm_y_c04) = eop_c04.get_pm(54195.0).unwrap();
    //     assert_abs_diff_eq!(pm_x_s, pm_x_c04, epsilon = 1.0e-9);
    //     assert_abs_diff_eq!(pm_y_s, pm_y_c04, epsilon = 1.0e-9);

    //     // Confirm ut1-utc are approximately equal
    //     let ut1_utc_s = eop_standard.get_ut1_utc(54195.0).unwrap();
    //     let ut1_utc_c04 = eop_c04.get_ut1_utc(54195.0).unwrap();
    //     assert_abs_diff_eq!(ut1_utc_s, ut1_utc_c04, epsilon = 1.0e-5);

    //     // Confirm LOD are approximately equal
    //     let lod_s = eop_standard.get_lod(54195.0).unwrap();
    //     let lod_c04 = eop_c04.get_lod(54195.0).unwrap();
    //     assert_abs_diff_eq!(lod_s, lod_c04, epsilon = 1.0e-4);

    //     // Confirm dX, and dY are not approximately equal even for the same file
    //     // let (dX_s, dY_s) = eop_standard.get_dxdy(54195.0).unwrap();
    //     // let (dX_c04, dY_c04) = eop_c04.get_dxdy(54195.0).unwrap();
    //     // assert_abs_diff_eq!(dX_s, dX_c04, epsilon = 1.0e-12)
    // }
}
