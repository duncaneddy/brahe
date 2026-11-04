/*!
 * IGS product file-name conventions.
 *
 * The IGS archives name precise products in one of two ways:
 *
 * - Long names, used from GPS week 2238 onward:
 *   `CCCVPPPTTT_YYYYDDDHHMM_LEN_SMP_CNT.FMT[.gz]`, for example
 *   `IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz`. The fields are the analysis
 *   center, version, campaign, solution type, start time, span, sampling
 *   interval, content type, and file format.
 * - Short names, used before week 2238: `cccwwwwd[_hh].ext[.Z|.gz]`, for
 *   example `igs21000.sp3.Z`. The fields are the analysis center and
 *   solution type (both encoded in a three-letter code), GPS week, day of
 *   week (`7` for a weekly product), the start hour of an ultra-rapid
 *   product, and the file extension.
 *
 * [`IGSProductFileName`] parses either form into typed fields and renders the
 * name exactly as it was published.
 */

use std::fmt;
use std::str::FromStr;

use crate::constants::GPS_ZERO;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// First GPS week for which the archives use long product file names.
pub(crate) const IGS_LONG_NAME_FIRST_WEEK: u32 = 2238;

const SECONDS_PER_DAY: f64 = 86400.0;
const SECONDS_PER_HOUR: f64 = 3600.0;
const SECONDS_PER_MINUTE: f64 = 60.0;
const SECONDS_PER_WEEK: f64 = 7.0 * SECONDS_PER_DAY;
const SECONDS_PER_YEAR_365: f64 = 365.0 * SECONDS_PER_DAY;

/// Span of a daily short-name product, in seconds.
const SHORT_DAILY_SPAN: f64 = SECONDS_PER_DAY;
/// Span of an ultra-rapid short-name product, in seconds.
const SHORT_ULTRA_SPAN: f64 = 2.0 * SECONDS_PER_DAY;
/// Observed-arc length that precedes the named time of an ultra-rapid
/// short-name product, in seconds. The named day and hour mark the start of
/// the predicted half, so coverage begins this long before it.
const SHORT_ULTRA_OBSERVED_LEAD: f64 = SECONDS_PER_DAY;

/// Modified Julian Date of 1970-01-01, the epoch of the day-count in
/// [`days_from_civil`].
const MJD_UNIX_EPOCH: i64 = 40587;

/// Short-name codes of centers whose rapid products are published under their
/// own code, as `(short code, center)`.
///
/// Codes absent from this table and from [`SHORT_ULTRA_CODES`] are final
/// products of the center named by the uppercased code. The IGS combination
/// codes (`igs`, `igr`, `igu`, `igl`, `igv`) follow the pre-week-2237 code
/// list at
/// <https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/precise-orbits-product>.
/// That page does not list the individual-center rapid and ultra-rapid codes
/// `cor`, `cou`, `emp`, `esr`, `esu`, `gfr` and `gfu`, so those entries are
/// unconfirmed. The SIO codes `sir` and `siu` come from the Garner archive
/// listing for GPS week 2100, <http://garner.ucsd.edu/pub/products/2100/>.
/// The GLONASS-only combinations `igl` and `igv` keep their own center codes
/// `IGL` and `IGV`.
const SHORT_RAPID_CODES: &[(&str, &str)] = &[
    ("igr", "IGS"),
    ("cor", "COD"),
    ("esr", "ESA"),
    ("emp", "EMR"),
    ("gfr", "GFZ"),
    ("sir", "SIO"),
];

/// Short-name codes of centers whose ultra-rapid products are published under
/// their own code, as `(short code, center)`. See [`SHORT_RAPID_CODES`].
const SHORT_ULTRA_CODES: &[(&str, &str)] = &[
    ("igu", "IGS"),
    ("igv", "IGV"),
    ("cou", "COD"),
    ("esu", "ESA"),
    ("gfu", "GFZ"),
    ("siu", "SIO"),
];

/// Product latency class of an IGS solution.
///
/// Long names carry the class as a three-letter code (`FIN`, `RAP`, `ULT`,
/// `PRD`). Short names carry it in the analysis-center code.
///
/// # Examples
///
/// ```
/// use brahe::igs::IGSSolutionType;
///
/// assert_eq!(IGSSolutionType::parse("FIN"), IGSSolutionType::Final);
/// assert_eq!(IGSSolutionType::UltraRapid.code(), "ULT");
/// assert_eq!(IGSSolutionType::parse("abc"), IGSSolutionType::Other("ABC".to_string()));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IGSSolutionType {
    /// Final solution (`FIN`).
    Final,
    /// Rapid solution (`RAP`).
    Rapid,
    /// Ultra-rapid solution (`ULT`).
    UltraRapid,
    /// Predicted solution (`PRD`).
    Predicted,
    /// Any other solution code, stored uppercased.
    Other(String),
}

impl IGSSolutionType {
    /// Parses a solution code.
    ///
    /// Matching is case-insensitive. Codes other than `FIN`, `RAP`, `ULT` and
    /// `PRD` are kept, uppercased, in [`IGSSolutionType::Other`].
    ///
    /// # Arguments
    /// * `code` - Three-letter solution code, for example `"FIN"`
    ///
    /// # Returns
    /// * `IGSSolutionType`: The parsed solution type
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSSolutionType;
    ///
    /// assert_eq!(IGSSolutionType::parse("rap"), IGSSolutionType::Rapid);
    /// assert_eq!(IGSSolutionType::parse("PRD"), IGSSolutionType::Predicted);
    /// ```
    pub fn parse(code: &str) -> Self {
        let upper = code.to_ascii_uppercase();
        match upper.as_str() {
            "FIN" => Self::Final,
            "RAP" => Self::Rapid,
            "ULT" => Self::UltraRapid,
            "PRD" => Self::Predicted,
            _ => Self::Other(upper),
        }
    }

    /// Returns the three-letter code used in long file names.
    ///
    /// # Returns
    /// * `&str`: The solution code, for example `"FIN"`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSSolutionType;
    ///
    /// assert_eq!(IGSSolutionType::Final.code(), "FIN");
    /// assert_eq!(IGSSolutionType::Other("XYZ".to_string()).code(), "XYZ");
    /// ```
    pub fn code(&self) -> &str {
        match self {
            Self::Final => "FIN",
            Self::Rapid => "RAP",
            Self::UltraRapid => "ULT",
            Self::Predicted => "PRD",
            Self::Other(code) => code,
        }
    }
}

/// Kind of data a product file holds.
///
/// Long names carry the content as a three-letter code (`ORB`, `CLK`, ...).
/// Short names imply it through the file extension.
///
/// # Examples
///
/// ```
/// use brahe::igs::IGSContentType;
///
/// assert_eq!(IGSContentType::parse("ORB"), IGSContentType::Orbit);
/// assert_eq!(IGSContentType::Ionosphere.code(), "GIM");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IGSContentType {
    /// Satellite orbits (`ORB`).
    Orbit,
    /// Satellite and station clocks (`CLK`).
    Clock,
    /// Earth rotation parameters (`ERP`).
    EarthRotation,
    /// Station coordinate solutions (`SOL`).
    Solution,
    /// Satellite attitude (`ATT`).
    Attitude,
    /// Signal biases (`BIA`).
    Bias,
    /// Global ionosphere maps (`GIM`).
    Ionosphere,
    /// Troposphere products (`TRO`).
    Troposphere,
    /// Solution summaries (`SUM`).
    Summary,
    /// Any other content code, stored uppercased.
    Other(String),
}

impl IGSContentType {
    /// Parses a content code.
    ///
    /// Matching is case-insensitive. Unknown codes are kept, uppercased, in
    /// [`IGSContentType::Other`].
    ///
    /// # Arguments
    /// * `code` - Three-letter content code, for example `"ORB"`
    ///
    /// # Returns
    /// * `IGSContentType`: The parsed content type
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSContentType;
    ///
    /// assert_eq!(IGSContentType::parse("erp"), IGSContentType::EarthRotation);
    /// assert_eq!(IGSContentType::parse("XYZ"), IGSContentType::Other("XYZ".to_string()));
    /// ```
    pub fn parse(code: &str) -> Self {
        let upper = code.to_ascii_uppercase();
        match upper.as_str() {
            "ORB" => Self::Orbit,
            "CLK" => Self::Clock,
            "ERP" => Self::EarthRotation,
            "SOL" => Self::Solution,
            "ATT" => Self::Attitude,
            "BIA" => Self::Bias,
            "GIM" => Self::Ionosphere,
            "TRO" => Self::Troposphere,
            "SUM" => Self::Summary,
            _ => Self::Other(upper),
        }
    }

    /// Returns the three-letter code used in long file names.
    ///
    /// # Returns
    /// * `&str`: The content code, for example `"ORB"`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSContentType;
    ///
    /// assert_eq!(IGSContentType::Clock.code(), "CLK");
    /// ```
    pub fn code(&self) -> &str {
        match self {
            Self::Orbit => "ORB",
            Self::Clock => "CLK",
            Self::EarthRotation => "ERP",
            Self::Solution => "SOL",
            Self::Attitude => "ATT",
            Self::Bias => "BIA",
            Self::Ionosphere => "GIM",
            Self::Troposphere => "TRO",
            Self::Summary => "SUM",
            Self::Other(code) => code,
        }
    }
}

/// Text format of a product file, taken from the file extension.
///
/// # Examples
///
/// ```
/// use brahe::igs::IGSFileFormat;
///
/// assert_eq!(IGSFileFormat::parse("sp3"), IGSFileFormat::SP3);
/// assert_eq!(IGSFileFormat::SNX.code(), "SNX");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum IGSFileFormat {
    /// SP3 precise orbit format.
    SP3,
    /// RINEX clock format.
    CLK,
    /// IGS Earth rotation parameter format.
    ERP,
    /// SINEX solution-independent exchange format.
    SNX,
    /// ORBEX orbit and attitude exchange format.
    OBX,
    /// Bias-SINEX signal bias format.
    BIA,
    /// IONEX ionosphere exchange format.
    INX,
    /// Troposphere SINEX format.
    TRO,
    /// Summary text.
    SUM,
    /// Any other format code, stored uppercased.
    Other(String),
}

impl IGSFileFormat {
    /// Parses a file-format code.
    ///
    /// Matching is case-insensitive. Unknown codes are kept, uppercased, in
    /// [`IGSFileFormat::Other`].
    ///
    /// # Arguments
    /// * `code` - File-format code, for example `"SP3"`
    ///
    /// # Returns
    /// * `IGSFileFormat`: The parsed file format
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSFileFormat;
    ///
    /// assert_eq!(IGSFileFormat::parse("CLK"), IGSFileFormat::CLK);
    /// assert_eq!(IGSFileFormat::parse("CLK_V2"), IGSFileFormat::Other("CLK_V2".to_string()));
    /// ```
    pub fn parse(code: &str) -> Self {
        let upper = code.to_ascii_uppercase();
        match upper.as_str() {
            "SP3" => Self::SP3,
            "CLK" => Self::CLK,
            "ERP" => Self::ERP,
            "SNX" => Self::SNX,
            "OBX" => Self::OBX,
            "BIA" => Self::BIA,
            "INX" => Self::INX,
            "TRO" => Self::TRO,
            "SUM" => Self::SUM,
            _ => Self::Other(upper),
        }
    }

    /// Returns the format code as written in long file names.
    ///
    /// # Returns
    /// * `&str`: The uppercase format code, for example `"SP3"`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSFileFormat;
    ///
    /// assert_eq!(IGSFileFormat::SP3.code(), "SP3");
    /// ```
    pub fn code(&self) -> &str {
        match self {
            Self::SP3 => "SP3",
            Self::CLK => "CLK",
            Self::ERP => "ERP",
            Self::SNX => "SNX",
            Self::OBX => "OBX",
            Self::BIA => "BIA",
            Self::INX => "INX",
            Self::TRO => "TRO",
            Self::SUM => "SUM",
            Self::Other(code) => code,
        }
    }
}

/// Compression applied to a published product file, from the trailing suffix.
///
/// # Examples
///
/// ```
/// use brahe::igs::{IGSCompression, IGSProductFileName};
///
/// let name = IGSProductFileName::parse("igs21000.sp3.Z").unwrap();
/// assert_eq!(name.compression(), IGSCompression::UnixCompress);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IGSCompression {
    /// Uncompressed.
    None,
    /// gzip (`.gz`).
    Gzip,
    /// Unix `compress` (`.Z`).
    UnixCompress,
}

/// Which of the two archive file-name conventions a name follows.
///
/// # Examples
///
/// ```
/// use brahe::igs::{IGSNameStyle, IGSProductFileName};
///
/// let name = IGSProductFileName::parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz").unwrap();
/// assert_eq!(name.name_style(), IGSNameStyle::Long);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IGSNameStyle {
    /// `CCCVPPPTTT_YYYYDDDHHMM_LEN_SMP_CNT.FMT` names.
    Long,
    /// `cccwwwwd[_hh].ext` names.
    Short,
}

/// A parsed IGS product file name.
///
/// `Display` renders the name exactly as it was parsed.
///
/// # Examples
///
/// ```
/// use brahe::igs::{IGSContentType, IGSProductFileName, IGSSolutionType};
///
/// let name = IGSProductFileName::parse("COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz").unwrap();
/// assert_eq!(name.center(), "COD");
/// assert_eq!(name.solution(), &IGSSolutionType::Final);
/// assert_eq!(name.content(), &IGSContentType::Orbit);
/// assert_eq!(name.gps_week(), 2340);
/// assert_eq!(name.span(), Some(86400.0));
/// assert_eq!(name.sampling(), Some(300.0));
/// assert!(name.is_sp3());
/// assert_eq!(name.to_string(), "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IGSProductFileName {
    name: String,
    style: IGSNameStyle,
    center: String,
    version: Option<u8>,
    campaign: Option<String>,
    solution: IGSSolutionType,
    start_epoch: Epoch,
    gps_week: u32,
    span: Option<f64>,
    sampling: Option<f64>,
    content: IGSContentType,
    format: IGSFileFormat,
    compression: IGSCompression,
}

impl IGSProductFileName {
    /// Parses a long or short IGS product file name.
    ///
    /// A name with five `_`-separated fields and a ten-character first field
    /// is read as a long name. Anything else is read as a short name. The
    /// trailing `.gz` or `.Z` is recognized as compression.
    ///
    /// For long names, span and sampling codes are a count followed by a unit
    /// (`S` seconds, `M` minutes, `H` hours, `D` days, `W` weeks, `Y` 365-day
    /// years). A code with any other unit leaves the span or sampling unset.
    /// The start time is read as GPS time.
    ///
    /// For short names, the analysis center and solution type come from the
    /// three-letter code, daily products span one day, ultra-rapid products
    /// span two days and start 24 hours before the named day and `_hh` hour
    /// (the named time is where the predicted half begins), and weekly
    /// products (day `7`) have no span and start at the beginning of the week. The sampling
    /// interval is not encoded.
    ///
    /// # Arguments
    /// * `name` - File name without directory components
    ///
    /// # Returns
    /// * `Ok(IGSProductFileName)`: The parsed name
    /// * `Err(BraheError)`: If the name matches neither convention or a field is out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSNameStyle, IGSProductFileName, IGSSolutionType};
    ///
    /// let name = IGSProductFileName::parse("igr21003.clk.Z").unwrap();
    /// assert_eq!(name.name_style(), IGSNameStyle::Short);
    /// assert_eq!(name.center(), "IGS");
    /// assert_eq!(name.solution(), &IGSSolutionType::Rapid);
    /// assert_eq!(name.gps_week(), 2100);
    ///
    /// assert!(IGSProductFileName::parse("readme.txt").is_err());
    /// ```
    pub fn parse(name: &str) -> Result<Self, BraheError> {
        let (body, compression) = if let Some(body) = name.strip_suffix(".gz") {
            (body, IGSCompression::Gzip)
        } else if let Some(body) = name.strip_suffix(".Z") {
            (body, IGSCompression::UnixCompress)
        } else {
            (name, IGSCompression::None)
        };

        if !name.is_ascii() {
            return Err(name_error(name, "name contains non-ASCII characters"));
        }
        let (stem, extension) = body
            .split_once('.')
            .ok_or_else(|| name_error(name, "missing file extension"))?;
        if extension.is_empty() {
            return Err(name_error(name, "empty file extension"));
        }

        let fields: Vec<&str> = stem.split('_').collect();
        if fields.len() == 5 && fields[0].len() == 10 {
            parse_long(name, &fields, extension, compression)
        } else {
            parse_short(name, stem, extension, compression)
        }
    }

    /// Returns which naming convention the name follows.
    ///
    /// # Returns
    /// * `IGSNameStyle`: `Long` or `Short`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSNameStyle, IGSProductFileName};
    ///
    /// assert_eq!(IGSProductFileName::parse("igs21000.sp3.Z").unwrap().name_style(), IGSNameStyle::Short);
    /// ```
    pub fn name_style(&self) -> IGSNameStyle {
        self.style
    }

    /// Returns the analysis-center code in uppercase.
    ///
    /// # Returns
    /// * `&str`: Three-character center code, for example `"COD"`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFileName;
    ///
    /// assert_eq!(IGSProductFileName::parse("cod21000.sp3").unwrap().center(), "COD");
    /// ```
    pub fn center(&self) -> &str {
        &self.center
    }

    /// Returns the version digit of a long name.
    ///
    /// # Returns
    /// * `Option<u8>`: The version, or `None` for short names
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFileName;
    ///
    /// let name = IGSProductFileName::parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz").unwrap();
    /// assert_eq!(name.version(), Some(0));
    /// ```
    pub fn version(&self) -> Option<u8> {
        self.version
    }

    /// Returns the campaign code of a long name.
    ///
    /// # Returns
    /// * `Option<&str>`: The campaign, for example `"OPS"`, or `None` for short names
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFileName;
    ///
    /// let name = IGSProductFileName::parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz").unwrap();
    /// assert_eq!(name.campaign(), Some("OPS"));
    /// ```
    pub fn campaign(&self) -> Option<&str> {
        self.campaign.as_deref()
    }

    /// Returns the solution type.
    ///
    /// # Returns
    /// * `&IGSSolutionType`: The solution type
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductFileName, IGSSolutionType};
    ///
    /// let name = IGSProductFileName::parse("igu21000_06.sp3.Z").unwrap();
    /// assert_eq!(name.solution(), &IGSSolutionType::UltraRapid);
    /// ```
    pub fn solution(&self) -> &IGSSolutionType {
        &self.solution
    }

    /// Returns the start of the product's coverage.
    ///
    /// For long names this is the start time in the name. For short
    /// ultra-rapid names (`igu`, `igv`, `siu`, ...) the named week, day and
    /// hour mark the start of the predicted half of the two-day file, so the
    /// coverage starts 24 hours earlier and can fall in the previous GPS week,
    /// whereas [`gps_week`](Self::gps_week) stays the named week.
    ///
    /// # Returns
    /// * `Epoch`: Start epoch in the GPS time system
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFileName;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let name = IGSProductFileName::parse("igs21000.sp3.Z").unwrap();
    /// assert_eq!(name.start_epoch(), Epoch::from_datetime(2020, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::GPS));
    ///
    /// let ultra = IGSProductFileName::parse("igu21000_06.sp3.Z").unwrap();
    /// assert_eq!(ultra.start_epoch(), Epoch::from_datetime(2020, 4, 4, 6, 0, 0.0, 0.0, TimeSystem::GPS));
    /// ```
    pub fn start_epoch(&self) -> Epoch {
        self.start_epoch
    }

    /// Returns the GPS week of the archive directory that holds the product.
    ///
    /// This is the week in the file name. It equals the week of
    /// [`start_epoch`](Self::start_epoch) except for short ultra-rapid names
    /// issued on the first day of a week, whose coverage starts in the
    /// previous week.
    ///
    /// # Returns
    /// * `u32`: GPS week number
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFileName;
    ///
    /// assert_eq!(IGSProductFileName::parse("igs21000.sp3.Z").unwrap().gps_week(), 2100);
    ///
    /// let ultra = IGSProductFileName::parse("igu21000_00.sp3.Z").unwrap();
    /// assert_eq!(ultra.gps_week(), 2100);
    /// assert_eq!(ultra.start_epoch().gps_date().0, 2099);
    /// ```
    pub fn gps_week(&self) -> u32 {
        self.gps_week
    }

    /// Returns the time covered by the file.
    ///
    /// # Returns
    /// * `Option<f64>`: Span in seconds, or `None` for short weekly products and long names with an unrecognized span unit
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFileName;
    ///
    /// assert_eq!(IGSProductFileName::parse("igu21000_06.sp3.Z").unwrap().span(), Some(172800.0));
    /// assert_eq!(IGSProductFileName::parse("igs21007.erp.Z").unwrap().span(), None);
    /// ```
    pub fn span(&self) -> Option<f64> {
        self.span
    }

    /// Returns the data sampling interval of a long name.
    ///
    /// # Returns
    /// * `Option<f64>`: Sampling interval in seconds, or `None` for short names and long names with an unrecognized unit
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFileName;
    ///
    /// let name = IGSProductFileName::parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz").unwrap();
    /// assert_eq!(name.sampling(), Some(900.0));
    /// ```
    pub fn sampling(&self) -> Option<f64> {
        self.sampling
    }

    /// Returns the kind of data the file holds.
    ///
    /// # Returns
    /// * `&IGSContentType`: The content type
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSContentType, IGSProductFileName};
    ///
    /// assert_eq!(IGSProductFileName::parse("igr21003.clk.Z").unwrap().content(), &IGSContentType::Clock);
    /// ```
    pub fn content(&self) -> &IGSContentType {
        &self.content
    }

    /// Returns the file format.
    ///
    /// # Returns
    /// * `&IGSFileFormat`: The file format
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSFileFormat, IGSProductFileName};
    ///
    /// assert_eq!(IGSProductFileName::parse("igs21000.sp3.Z").unwrap().format(), &IGSFileFormat::SP3);
    /// ```
    pub fn format(&self) -> &IGSFileFormat {
        &self.format
    }

    /// Returns the compression applied to the published file.
    ///
    /// # Returns
    /// * `IGSCompression`: Compression from the trailing `.gz` or `.Z`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSCompression, IGSProductFileName};
    ///
    /// assert_eq!(IGSProductFileName::parse("cod21000.sp3").unwrap().compression(), IGSCompression::None);
    /// ```
    pub fn compression(&self) -> IGSCompression {
        self.compression
    }

    /// Returns whether the file is an SP3 orbit file.
    ///
    /// # Returns
    /// * `bool`: `true` if the file format is SP3
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFileName;
    ///
    /// assert!(IGSProductFileName::parse("igs21000.sp3.Z").unwrap().is_sp3());
    /// assert!(!IGSProductFileName::parse("igs21000.erp.Z").unwrap().is_sp3());
    /// ```
    pub fn is_sp3(&self) -> bool {
        self.format == IGSFileFormat::SP3
    }
}

impl fmt::Display for IGSProductFileName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl FromStr for IGSProductFileName {
    type Err = BraheError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

fn name_error(name: &str, detail: &str) -> BraheError {
    BraheError::ParseError(format!("IGS file name '{}': {}", name, detail))
}

/// Parses a fixed-width run of ASCII digits.
fn parse_digits(name: &str, field: &str, text: &str) -> Result<u32, BraheError> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(name_error(
            name,
            &format!("{} '{}' is not numeric", field, text),
        ));
    }
    text.parse::<u32>()
        .map_err(|_| name_error(name, &format!("{} '{}' is out of range", field, text)))
}

/// Days since 1970-01-01 of a proleptic Gregorian calendar date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// GPS week containing a calendar day, from integer day counts.
fn gps_week_of_day(name: &str, year: u32, day_of_year: u32) -> Result<u32, BraheError> {
    let mjd = days_from_civil(i64::from(year), 1, 1) + MJD_UNIX_EPOCH + i64::from(day_of_year) - 1;
    let days_since_gps = mjd - GPS_ZERO as i64;
    if days_since_gps < 0 {
        return Err(name_error(name, "start time precedes the GPS epoch"));
    }
    Ok((days_since_gps / 7) as u32)
}

/// Converts a long-name span or sampling code such as `01D` to seconds.
fn parse_duration_code(name: &str, field: &str, code: &str) -> Result<Option<f64>, BraheError> {
    let Some(unit) = code.chars().last() else {
        return Err(name_error(name, &format!("empty {} code", field)));
    };
    let count = &code[..code.len() - unit.len_utf8()];
    if !unit.is_ascii_alphabetic() || count.is_empty() {
        return Err(name_error(
            name,
            &format!("{} code '{}' is not a count and unit", field, code),
        ));
    }
    let count = f64::from(parse_digits(name, field, count)?);
    let unit_seconds = match unit {
        'S' => 1.0,
        'M' => SECONDS_PER_MINUTE,
        'H' => SECONDS_PER_HOUR,
        'D' => SECONDS_PER_DAY,
        'W' => SECONDS_PER_WEEK,
        'Y' => SECONDS_PER_YEAR_365,
        _ => return Ok(None),
    };
    Ok(Some(count * unit_seconds))
}

fn parse_long(
    name: &str,
    fields: &[&str],
    extension: &str,
    compression: IGSCompression,
) -> Result<IGSProductFileName, BraheError> {
    let head = fields[0];
    if !head.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err(name_error(
            name,
            "center, version, campaign and solution must be alphanumeric",
        ));
    }
    let center = head[0..3].to_ascii_uppercase();
    let version = head.as_bytes()[3];
    if !version.is_ascii_digit() {
        return Err(name_error(name, "version must be a digit"));
    }
    let campaign = head[4..7].to_ascii_uppercase();
    let solution = IGSSolutionType::parse(&head[7..10]);

    let start = fields[1];
    if start.len() != 11 {
        return Err(name_error(
            name,
            &format!("start time '{}' must be YYYYDDDHHMM", start),
        ));
    }
    let year = parse_digits(name, "year", &start[0..4])?;
    let day_of_year = parse_digits(name, "day of year", &start[4..7])?;
    let hour = parse_digits(name, "hour", &start[7..9])?;
    let minute = parse_digits(name, "minute", &start[9..11])?;
    let days_in_year = if is_leap_year(year) { 366 } else { 365 };
    if day_of_year < 1 || day_of_year > days_in_year {
        return Err(name_error(
            name,
            &format!("day of year {} is out of range for {}", day_of_year, year),
        ));
    }
    if hour > 23 || minute > 59 {
        return Err(name_error(
            name,
            &format!("start time {:02}:{:02} is out of range", hour, minute),
        ));
    }
    let gps_week = gps_week_of_day(name, year, day_of_year)?;
    let start_epoch = Epoch::from_datetime(
        year,
        1,
        1,
        hour as u8,
        minute as u8,
        0.0,
        0.0,
        TimeSystem::GPS,
    ) + f64::from(day_of_year - 1) * SECONDS_PER_DAY;

    let span = parse_duration_code(name, "span", fields[2])?;
    let sampling = parse_duration_code(name, "sampling", fields[3])?;

    if fields[4].len() != 3 || !fields[4].bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err(name_error(
            name,
            &format!(
                "content code '{}' must be three alphanumeric characters",
                fields[4]
            ),
        ));
    }

    Ok(IGSProductFileName {
        name: name.to_string(),
        style: IGSNameStyle::Long,
        center,
        version: Some(version - b'0'),
        campaign: Some(campaign),
        solution,
        start_epoch,
        gps_week,
        span,
        sampling,
        content: IGSContentType::parse(fields[4]),
        format: IGSFileFormat::parse(extension),
        compression,
    })
}

/// Maps a short-name extension to its content type and file format.
fn short_content_and_format(extension: &str) -> (IGSContentType, IGSFileFormat) {
    match extension.to_ascii_lowercase().as_str() {
        "sp3" | "eph" => (IGSContentType::Orbit, IGSFileFormat::SP3),
        "clk" | "clk_30s" | "clk_05s" => (IGSContentType::Clock, IGSFileFormat::CLK),
        "erp" => (IGSContentType::EarthRotation, IGSFileFormat::ERP),
        "snx" => (IGSContentType::Solution, IGSFileFormat::SNX),
        "sum" => (IGSContentType::Summary, IGSFileFormat::SUM),
        "tro" => (IGSContentType::Troposphere, IGSFileFormat::TRO),
        _ => (
            IGSContentType::Other(extension.to_ascii_uppercase()),
            IGSFileFormat::Other(extension.to_ascii_uppercase()),
        ),
    }
}

/// Resolves a three-letter short-name code to its center and solution type.
fn short_center_and_solution(code: &str) -> (String, IGSSolutionType) {
    if let Some((_, center)) = SHORT_RAPID_CODES.iter().find(|(c, _)| *c == code) {
        return (center.to_string(), IGSSolutionType::Rapid);
    }
    if let Some((_, center)) = SHORT_ULTRA_CODES.iter().find(|(c, _)| *c == code) {
        return (center.to_string(), IGSSolutionType::UltraRapid);
    }
    (code.to_ascii_uppercase(), IGSSolutionType::Final)
}

/// Three-letter short-name code under which `center` publishes products of
/// `solution`.
///
/// This is the inverse of [`short_center_and_solution`]: a rapid or
/// ultra-rapid product uses the center's code from [`SHORT_RAPID_CODES`] or
/// [`SHORT_ULTRA_CODES`], and every other product uses the lowercased center.
///
/// # Arguments
/// * `center` - Analysis center code, in any case
/// * `solution` - Solution type of the product
///
/// # Returns
/// * `String`: Lowercase three-letter code, for example `"igr"` for IGS rapid products
pub(crate) fn short_code(center: &str, solution: &IGSSolutionType) -> String {
    let table = match solution {
        IGSSolutionType::Rapid => SHORT_RAPID_CODES,
        IGSSolutionType::UltraRapid => SHORT_ULTRA_CODES,
        _ => &[],
    };
    table
        .iter()
        .find(|(_, c)| c.eq_ignore_ascii_case(center))
        .map(|(code, _)| code.to_string())
        .unwrap_or_else(|| center.to_ascii_lowercase())
}

fn parse_short(
    name: &str,
    stem: &str,
    extension: &str,
    compression: IGSCompression,
) -> Result<IGSProductFileName, BraheError> {
    let (code_week_day, hour_text) = match stem.split_once('_') {
        Some((head, hour)) => (head, Some(hour)),
        None => (stem, None),
    };
    if code_week_day.len() != 8 {
        return Err(name_error(
            name,
            "matches neither the long nor the short product naming convention",
        ));
    }
    let code = code_week_day[0..3].to_ascii_lowercase();
    if !code.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err(name_error(name, "center code must be alphanumeric"));
    }
    let gps_week = parse_digits(name, "GPS week", &code_week_day[3..7])?;
    let day = parse_digits(name, "day of week", &code_week_day[7..8])?;
    if day > 7 {
        return Err(name_error(
            name,
            &format!("day of week {} must be 0 to 7", day),
        ));
    }

    let (center, solution) = short_center_and_solution(&code);
    let ultra = solution == IGSSolutionType::UltraRapid;
    let hour = match hour_text {
        None => 0,
        Some(text) if ultra && text.len() == 2 => {
            let hour = parse_digits(name, "hour", text)?;
            if hour > 23 {
                return Err(name_error(name, &format!("hour {} is out of range", hour)));
            }
            hour
        }
        Some(text) => {
            return Err(name_error(
                name,
                &format!("unexpected '_{}' after the day of week", text),
            ));
        }
    };

    let weekly = day == 7;
    let span = if weekly {
        None
    } else if ultra {
        Some(SHORT_ULTRA_SPAN)
    } else {
        Some(SHORT_DAILY_SPAN)
    };
    let day_offset = if weekly { 0 } else { day };
    let named_seconds =
        f64::from(day_offset) * SECONDS_PER_DAY + f64::from(hour) * SECONDS_PER_HOUR;
    let coverage_offset = if ultra && !weekly {
        SHORT_ULTRA_OBSERVED_LEAD
    } else {
        0.0
    };
    let start_epoch = Epoch::from_gps_date(gps_week, named_seconds) - coverage_offset;
    let (content, format) = short_content_and_format(extension);

    Ok(IGSProductFileName {
        name: name.to_string(),
        style: IGSNameStyle::Short,
        center,
        version: None,
        campaign: None,
        solution,
        start_epoch,
        gps_week,
        span,
        sampling: None,
        content,
        format,
        compression,
    })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use rstest::rstest;
    use serial_test::parallel;

    fn gps(y: u32, mo: u8, d: u8, h: u8) -> Epoch {
        Epoch::from_datetime(y, mo, d, h, 0, 0.0, 0.0, TimeSystem::GPS)
    }

    #[rstest]
    #[case(
        "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz",
        IGSNameStyle::Long,
        "IGS",
        IGSSolutionType::Final,
        gps(2024, 11, 10, 0),
        2340
    )]
    #[case(
        "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz",
        IGSNameStyle::Long,
        "COD",
        IGSSolutionType::Final,
        gps(2024, 11, 10, 0),
        2340
    )]
    #[case(
        "IGS0OPSULT_20243150600_02D_15M_ORB.SP3.gz",
        IGSNameStyle::Long,
        "IGS",
        IGSSolutionType::UltraRapid,
        gps(2024, 11, 10, 6),
        2340
    )]
    #[case(
        "COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz",
        IGSNameStyle::Long,
        "COD",
        IGSSolutionType::Final,
        gps(2024, 11, 10, 0),
        2340
    )]
    #[case(
        "IGS0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
        IGSNameStyle::Long,
        "IGS",
        IGSSolutionType::Final,
        gps(2024, 11, 10, 0),
        2340
    )]
    #[case(
        "igs21000.sp3.Z",
        IGSNameStyle::Short,
        "IGS",
        IGSSolutionType::Final,
        gps(2020, 4, 5, 0),
        2100
    )]
    #[case(
        "igr21003.clk.Z",
        IGSNameStyle::Short,
        "IGS",
        IGSSolutionType::Rapid,
        gps(2020, 4, 8, 0),
        2100
    )]
    #[case(
        "igu21000_06.sp3.Z",
        IGSNameStyle::Short,
        "IGS",
        IGSSolutionType::UltraRapid,
        gps(2020, 4, 4, 6),
        2100
    )]
    #[case(
        "igs21007.erp.Z",
        IGSNameStyle::Short,
        "IGS",
        IGSSolutionType::Final,
        gps(2020, 4, 5, 0),
        2100
    )]
    #[case(
        "cod21000.sp3",
        IGSNameStyle::Short,
        "COD",
        IGSSolutionType::Final,
        gps(2020, 4, 5, 0),
        2100
    )]
    #[parallel]
    fn test_igs_product_file_name_parse_identity(
        #[case] input: &str,
        #[case] style: IGSNameStyle,
        #[case] center: &str,
        #[case] solution: IGSSolutionType,
        #[case] start: Epoch,
        #[case] week: u32,
    ) {
        let name = IGSProductFileName::parse(input).unwrap();
        assert_eq!(name.name_style(), style);
        assert_eq!(name.center(), center);
        assert_eq!(name.solution(), &solution);
        assert_eq!(name.start_epoch(), start);
        assert_eq!(name.gps_week(), week);
        if solution != IGSSolutionType::UltraRapid {
            assert_eq!(name.start_epoch().gps_date().0, week);
        }
        assert_eq!(name.to_string(), input);
        assert_eq!(input.parse::<IGSProductFileName>().unwrap(), name);
        match style {
            IGSNameStyle::Long => {
                assert_eq!(name.version(), Some(0));
                assert_eq!(name.campaign(), Some("OPS"));
            }
            IGSNameStyle::Short => {
                assert_eq!(name.version(), None);
                assert_eq!(name.campaign(), None);
            }
        }
    }

    #[rstest]
    #[case(
        "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz",
        Some(86400.0),
        Some(900.0),
        IGSContentType::Orbit,
        IGSFileFormat::SP3,
        IGSCompression::Gzip
    )]
    #[case(
        "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz",
        Some(86400.0),
        Some(300.0),
        IGSContentType::Orbit,
        IGSFileFormat::SP3,
        IGSCompression::Gzip
    )]
    #[case(
        "IGS0OPSULT_20243150600_02D_15M_ORB.SP3.gz",
        Some(172800.0),
        Some(900.0),
        IGSContentType::Orbit,
        IGSFileFormat::SP3,
        IGSCompression::Gzip
    )]
    #[case(
        "COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz",
        Some(86400.0),
        Some(30.0),
        IGSContentType::Clock,
        IGSFileFormat::CLK,
        IGSCompression::Gzip
    )]
    #[case(
        "IGS0OPSFIN_20243150000_07D_01D_ERP.ERP.gz",
        Some(604800.0),
        Some(86400.0),
        IGSContentType::EarthRotation,
        IGSFileFormat::ERP,
        IGSCompression::Gzip
    )]
    #[case(
        "igs21000.sp3.Z",
        Some(86400.0),
        None,
        IGSContentType::Orbit,
        IGSFileFormat::SP3,
        IGSCompression::UnixCompress
    )]
    #[case(
        "igr21003.clk.Z",
        Some(86400.0),
        None,
        IGSContentType::Clock,
        IGSFileFormat::CLK,
        IGSCompression::UnixCompress
    )]
    #[case(
        "igu21000_06.sp3.Z",
        Some(172800.0),
        None,
        IGSContentType::Orbit,
        IGSFileFormat::SP3,
        IGSCompression::UnixCompress
    )]
    #[case(
        "igs21007.erp.Z",
        None,
        None,
        IGSContentType::EarthRotation,
        IGSFileFormat::ERP,
        IGSCompression::UnixCompress
    )]
    #[case(
        "cod21000.sp3",
        Some(86400.0),
        None,
        IGSContentType::Orbit,
        IGSFileFormat::SP3,
        IGSCompression::None
    )]
    #[parallel]
    fn test_igs_product_file_name_parse_product(
        #[case] input: &str,
        #[case] span: Option<f64>,
        #[case] sampling: Option<f64>,
        #[case] content: IGSContentType,
        #[case] format: IGSFileFormat,
        #[case] compression: IGSCompression,
    ) {
        let name = IGSProductFileName::parse(input).unwrap();
        assert_eq!(name.span(), span);
        assert_eq!(name.sampling(), sampling);
        assert_eq!(name.content(), &content);
        assert_eq!(name.format(), &format);
        assert_eq!(name.compression(), compression);
        assert_eq!(name.is_sp3(), format == IGSFileFormat::SP3);
    }

    #[rstest]
    #[case("readme.txt")]
    #[case("IGS0OPSFIN_2024315_01D_15M_ORB.SP3")]
    #[case("igs2100.sp3")]
    #[case("igs21008.sp3")]
    #[case("igs21000")]
    #[case("igs21000.")]
    #[case("IGS0OPSFIN_20243670000_01D_15M_ORB.SP3")]
    #[case("IGS0OPSFIN_20253660000_01D_15M_ORB.SP3")]
    #[case("IGS0OPSFIN_20243152400_01D_15M_ORB.SP3")]
    #[case("IGS0OPSFIN_20243150060_01D_15M_ORB.SP3.gz")]
    #[case("IGS0OPSFIN_19791000000_01D_15M_ORB.SP3")]
    #[case("IGSXOPSFIN_20243150000_01D_15M_ORB.SP3")]
    #[case("IGS0OPSFIN_20243150000_D_15M_ORB.SP3")]
    #[case("igs21000_06.sp3")]
    #[case("igu21000_24.sp3")]
    #[case("igu21000_6.sp3")]
    #[parallel]
    fn test_igs_product_file_name_parse_errors(#[case] input: &str) {
        assert!(
            IGSProductFileName::parse(input).is_err(),
            "expected '{}' to be rejected",
            input
        );
    }

    #[rstest]
    #[case("IGS0OPSFIN_2024A150000_01D_15M_ORB.SP3")]
    #[case("IGS0OPSFIN_20243150000__15M_ORB.SP3")]
    #[case("IGS0OPSFIN_20243150000_01D__ORB.SP3")]
    #[case("IGS-OPSFIN_20243150000_01D_15M_ORB.SP3")]
    #[case("IGS0OPSFIN_20243150000_01D_15M_OR.SP3")]
    #[case("IGS0OPSFIN_20243150000_01D_15M_OR-.SP3")]
    #[case("ig-21000.sp3")]
    #[parallel]
    fn test_igs_product_file_name_malformed_fields(#[case] input: &str) {
        assert!(
            IGSProductFileName::parse(input).is_err(),
            "expected '{}' to be rejected",
            input
        );
    }

    #[rstest]
    #[case("cor21000.sp3.Z", "COD", IGSSolutionType::Rapid)]
    #[case("cou21000_12.sp3.Z", "COD", IGSSolutionType::UltraRapid)]
    #[case("esr21000.sp3.Z", "ESA", IGSSolutionType::Rapid)]
    #[case("esu21000_18.sp3.Z", "ESA", IGSSolutionType::UltraRapid)]
    #[case("emr21000.sp3.Z", "EMR", IGSSolutionType::Final)]
    #[case("emp21000.sp3.Z", "EMR", IGSSolutionType::Rapid)]
    #[case("gfr21000.sp3.Z", "GFZ", IGSSolutionType::Rapid)]
    #[case("gfu21000_00.sp3.Z", "GFZ", IGSSolutionType::UltraRapid)]
    #[case("jpl21000.sp3.Z", "JPL", IGSSolutionType::Final)]
    #[case("igl21000.sp3.Z", "IGL", IGSSolutionType::Final)]
    #[case("igv21000_06.sp3.Z", "IGV", IGSSolutionType::UltraRapid)]
    #[case("siu21000_02.sp3.Z", "SIO", IGSSolutionType::UltraRapid)]
    #[case("sir21003.erp.Z", "SIO", IGSSolutionType::Rapid)]
    #[case("cof21000.eph.Z", "COF", IGSSolutionType::Final)]
    #[parallel]
    fn test_igs_product_file_name_short_center_codes(
        #[case] input: &str,
        #[case] center: &str,
        #[case] solution: IGSSolutionType,
    ) {
        let name = IGSProductFileName::parse(input).unwrap();
        assert_eq!(name.center(), center);
        assert_eq!(name.solution(), &solution);
    }

    #[rstest]
    #[case("01S", Some(1.0))]
    #[case("15M", Some(900.0))]
    #[case("06H", Some(21600.0))]
    #[case("01D", Some(86400.0))]
    #[case("02W", Some(1209600.0))]
    #[case("01Y", Some(31536000.0))]
    #[case("00U", None)]
    #[parallel]
    fn test_igs_product_file_name_duration_units(
        #[case] code: &str,
        #[case] expected: Option<f64>,
    ) {
        let input = format!("IGS0OPSFIN_20243150000_{}_{}_ORB.SP3", code, code);
        let name = IGSProductFileName::parse(&input).unwrap();
        assert_eq!(name.span(), expected);
        assert_eq!(name.sampling(), expected);
    }

    #[rstest]
    #[case("igs21000.clk_30s.Z", IGSContentType::Clock, IGSFileFormat::CLK)]
    #[case("igs21000.clk_05s.Z", IGSContentType::Clock, IGSFileFormat::CLK)]
    #[case("igs21000.snx.Z", IGSContentType::Solution, IGSFileFormat::SNX)]
    #[case("igs21000.sum", IGSContentType::Summary, IGSFileFormat::SUM)]
    #[case("cof21000.tro.Z", IGSContentType::Troposphere, IGSFileFormat::TRO)]
    #[case("igs21000.xyz", IGSContentType::Other("XYZ".to_string()), IGSFileFormat::Other("XYZ".to_string()))]
    #[parallel]
    fn test_igs_product_file_name_short_extensions(
        #[case] input: &str,
        #[case] content: IGSContentType,
        #[case] format: IGSFileFormat,
    ) {
        let name = IGSProductFileName::parse(input).unwrap();
        assert_eq!(name.content(), &content);
        assert_eq!(name.format(), &format);
        assert!(!name.is_sp3());
        assert_eq!(name.to_string(), input);
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_long_other_codes() {
        let name = IGSProductFileName::parse("WUM0MGXFIN_20243150000_01D_05M_ABC.XYZ").unwrap();
        assert_eq!(name.campaign(), Some("MGX"));
        assert_eq!(name.content(), &IGSContentType::Other("ABC".to_string()));
        assert_eq!(name.format(), &IGSFileFormat::Other("XYZ".to_string()));
        assert_eq!(name.compression(), IGSCompression::None);
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_long_week_boundaries() {
        let sunday = IGSProductFileName::parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3").unwrap();
        let saturday = IGSProductFileName::parse("IGS0OPSFIN_20243142359_01D_15M_ORB.SP3").unwrap();
        assert_eq!(sunday.gps_week(), 2340);
        assert_eq!(saturday.gps_week(), 2339);
        assert_eq!(saturday.start_epoch().gps_date().0, 2339);
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_long_leap_day() {
        let name = IGSProductFileName::parse("IGS0OPSFIN_20243660000_01D_15M_ORB.SP3").unwrap();
        assert_eq!(name.start_epoch(), gps(2024, 12, 31, 0));
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_igv_ultra_rapid_span() {
        let name = IGSProductFileName::parse("igv21000_06.sp3").unwrap();
        assert_eq!(name.start_epoch(), gps(2020, 4, 4, 6));
        assert_eq!(name.span(), Some(172800.0));
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_short_ultra_rapid_week_boundary() {
        let name = IGSProductFileName::parse("igu21000_00.sp3.Z").unwrap();
        assert_eq!(name.gps_week(), 2100);
        assert_eq!(name.start_epoch(), gps(2020, 4, 4, 0));
        assert_eq!(name.start_epoch().gps_date().0, 2099);
        assert_eq!(name.span(), Some(172800.0));
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_siu_ultra_rapid() {
        let name = IGSProductFileName::parse("siu21000_02.sp3.Z").unwrap();
        assert_eq!(name.start_epoch(), gps(2020, 4, 4, 2));
        assert_eq!(name.span(), Some(172800.0));
        assert_eq!(name.compression(), IGSCompression::UnixCompress);
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_sir_earth_rotation() {
        let name = IGSProductFileName::parse("sir21003.erp.Z").unwrap();
        assert_eq!(name.content(), &IGSContentType::EarthRotation);
        assert_eq!(name.format(), &IGSFileFormat::ERP);
        assert_eq!(name.start_epoch(), gps(2020, 4, 8, 0));
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_eph_extension_is_sp3() {
        let name = IGSProductFileName::parse("cof21000.eph.Z").unwrap();
        assert_eq!(name.content(), &IGSContentType::Orbit);
        assert_eq!(name.format(), &IGSFileFormat::SP3);
        assert!(name.is_sp3());
        assert_eq!(name.to_string(), "cof21000.eph.Z");
    }

    #[test]
    #[parallel]
    fn test_igs_product_file_name_compression_suffix_by_style() {
        let long = IGSProductFileName::parse("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.Z").unwrap();
        assert_eq!(long.compression(), IGSCompression::UnixCompress);
        assert_eq!(long.format(), &IGSFileFormat::SP3);
        assert_eq!(long.to_string(), "IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.Z");

        let short = IGSProductFileName::parse("igs21000.sp3.gz").unwrap();
        assert_eq!(short.compression(), IGSCompression::Gzip);
        assert_eq!(short.format(), &IGSFileFormat::SP3);
        assert_eq!(short.to_string(), "igs21000.sp3.gz");
    }

    #[rstest]
    #[case("igs21000é.sp3")]
    #[case("IGS0OPSFIN_20243150000_01D_15M_ORB.SP3.gz\u{00e9}")]
    #[case("IGé0OPSFIN_20243150000_01D_15M_ORB.SP3")]
    #[parallel]
    fn test_igs_product_file_name_non_ascii(#[case] input: &str) {
        assert!(IGSProductFileName::parse(input).is_err());
    }

    #[test]
    #[parallel]
    fn test_igs_solution_content_format_codes() {
        for code in ["FIN", "RAP", "ULT", "PRD", "XYZ"] {
            assert_eq!(IGSSolutionType::parse(code).code(), code);
        }
        for code in [
            "ORB", "CLK", "ERP", "SOL", "ATT", "BIA", "GIM", "TRO", "SUM", "XYZ",
        ] {
            assert_eq!(IGSContentType::parse(code).code(), code);
        }
        for code in [
            "SP3", "CLK", "ERP", "SNX", "OBX", "BIA", "INX", "TRO", "SUM", "XYZ",
        ] {
            assert_eq!(IGSFileFormat::parse(code).code(), code);
        }
        assert_eq!(IGSSolutionType::parse("fin"), IGSSolutionType::Final);
    }

    #[rstest]
    #[case("IGS", IGSSolutionType::Final, "igs")]
    #[case("igs", IGSSolutionType::Rapid, "igr")]
    #[case("IGS", IGSSolutionType::UltraRapid, "igu")]
    #[case("COD", IGSSolutionType::Final, "cod")]
    #[case("cod", IGSSolutionType::Rapid, "cor")]
    #[case("EMR", IGSSolutionType::Rapid, "emp")]
    #[case("JPL", IGSSolutionType::Rapid, "jpl")]
    #[case("COD", IGSSolutionType::Predicted, "cod")]
    #[parallel]
    fn test_igs_short_code(
        #[case] center: &str,
        #[case] solution: IGSSolutionType,
        #[case] expected: &str,
    ) {
        let code = short_code(center, &solution);
        assert_eq!(code, expected);
        if expected != "jpl" && solution != IGSSolutionType::Predicted {
            let (parsed_center, parsed_solution) = short_center_and_solution(&code);
            assert!(parsed_center.eq_ignore_ascii_case(center));
            assert_eq!(parsed_solution, solution);
        }
    }
}
